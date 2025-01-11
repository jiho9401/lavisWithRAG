import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np

from torch.cuda.amp import autocast as autocast
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    AutoTokenizer,
    OPTForCausalLM,
)

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM as LavisOPTForCausalLM
from lavis.models.blip2_models.modeling_opt import OPTConfig


@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 + OPT 모델 (원본).
    RAG 기능을 함께 사용하기 위해, 아래에 DPR, FAISS 관련 코드를 추가로 포함합니다.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        # -- RAG 관련 파라미터 추가 --
        retriever_model_name="facebook/dpr-ctx_encoder-multiset-base",
        use_fp16=True,
    ):
        """
        BLIP-2 OPT 초기화 + DPR/Faiss RAG 관련 멤버 추가
        """
        super().__init__()

        # 1) BLIP-2 (Vision + Qformer + OPT) 초기화
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        # 이하 원본 Qformer 관련 코드
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = LavisOPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False

        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        # 2) RAG를 위한 DPR & FAISS 초기화
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.ctx_encoder = DPRContextEncoder.from_pretrained(
                retriever_model_name, torch_dtype=torch.float16
            ).eval().cuda()
        else:
            self.ctx_encoder = DPRContextEncoder.from_pretrained(
                retriever_model_name
            ).eval().cuda()

        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(retriever_model_name)

        # FAISS CPU Index
        self.faiss_index = None
        self.documents = []

        # BLIP-2 Query Embedding → DPR 차원(기본 768) 투영 레이어
        # BLIP-2의 최종 Query Embedding이 self.opt_proj 이전까지는 768일 수도 있지만,
        # 필요 시 별도 Projection이 가능하다고 가정.
        # 여기서는 2048이나 2560 등, 실제로 원하는 dimension으로 설정 가능.
        # 일단 2560->768이라고 가정하면:
        self.query_projection = nn.Linear(self.opt_proj.in_features, 768)
        if self.use_fp16:
            self.query_projection = self.query_projection.half()
        self.query_projection.eval().cuda()
        for p in self.query_projection.parameters():
            p.requires_grad = False

    def forward(self, samples):
        """
        원본 BLIP-2 forward (학습/추론 시 사용).
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"
        text = [t + "\n" for t in samples["text_input"]]
        opt_tokens = self.opt_tokenizer(
            text, return_tensors="pt", padding="longest", truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100

        empty_targets = torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        원본 BLIP-2 generate 함수 (이미지 + 프롬프트 → 텍스트 캡셔닝/질의응답).
        """
        image = samples["image"]
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt
            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(image.device)
            input_ids = opt_tokens.input_ids
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            outputs = self.opt_model.generate(
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

    # --------------------------------------------------
    # (아래부터 RAG 기능 추가)
    # --------------------------------------------------

    def build_faiss_index(self, documents, save_path=None):
        """
        DPR로 문서 임베딩 → CPU 상의 Faiss Index 생성
        """
        ctx_embeddings = []
        for doc in documents:
            inputs = self.ctx_tokenizer(
                doc, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=(self.ctx_encoder.dtype == torch.float16)):
                    embedding = self.ctx_encoder(**inputs).pooler_output

            # CPU로 내리고 float32 변환
            embedding_np = embedding.cpu().numpy().astype(np.float32)
            ctx_embeddings.append(embedding_np)

        ctx_embeddings = np.vstack(ctx_embeddings)
        dimension = ctx_embeddings.shape[1]

        self.faiss_index = faiss.IndexFlatL2(dimension)  # CPU index
        self.faiss_index.add(ctx_embeddings)
        self.documents = documents

        if save_path is not None:
            np.save(save_path, ctx_embeddings)

    @torch.no_grad()
    def get_query_embedding(self, image: torch.Tensor):
        """
        BLIP-2로부터 이미지 임베딩을 추출한 뒤
        DPR(768차원)에 맞도록 별도 Projection (query_projection).
        """
        with torch.amp.autocast("cuda", enabled=(self.ln_vision.weight.dtype == torch.float16)):
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            # BLIP-2 최종 출력(OPT projection 전 단계) 사용 예시
            # 만약 self.opt_proj 결과(OPT 임베딩)를 쓰고 싶다면 아래를 바꾸면 됨.
            q_hidden = query_output.last_hidden_state.mean(dim=1)

        # query_projection 레이어가 half이면
        if self.query_projection.weight.dtype == torch.float16:
            q_hidden = q_hidden.half()
        else:
            q_hidden = q_hidden.float()

        projected = self.query_projection(q_hidden)

        # FAISS가 float32를 요구하므로 변환
        projected_np = projected.detach().cpu().numpy().astype(np.float32)
        return projected_np

    @torch.no_grad()
    def generate_rag(self, image, documents, query, k=3, **generate_kwargs):
        """
        1) (선택) faiss_index가 없으면 생성
        2) 이미지 → query 임베딩
        3) faiss로 문서 검색
        4) OPT로 최종 텍스트 생성
        """
        if self.faiss_index is None:
            self.build_faiss_index(documents)

        q_embedding = self.get_query_embedding(image)
        distances, indices = self.faiss_index.search(q_embedding, k=k)
        retrieved_docs = [self.documents[idx] for idx in indices[0]]

        # 컨텍스트 결합
        combined_context = " ".join(retrieved_docs)
        input_text = f"Context: {combined_context}\nQuery: {query}"

        opt_inputs = self.opt_tokenizer(
            input_text, return_tensors="pt", truncation=True
        ).cuda()

        with torch.amp.autocast("cuda", enabled=(self.opt_model.config.torch_dtype == torch.float16)):
            output = self.opt_model.generate(**opt_inputs, **generate_kwargs)

        output_text = self.opt_tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

    @classmethod
    def from_config(cls, cfg):
        """
        config를 통해 모델 초기화 (원본 BLIP-2)
        """
        vit_model = cfg.get("vit_model","eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        # RAG 추가 파라미터 예시
        retriever_model_name = cfg.get("retriever_model_name", "facebook/dpr-ctx_encoder-multiset-base")
        use_fp16 = cfg.get("use_fp16", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            retriever_model_name=retriever_model_name,
            use_fp16=use_fp16,
        )
        model.load_checkpoint_from_config(cfg)
        return model
