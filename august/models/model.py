from typing import Optional, Tuple, Dict, List
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from lora.injector import inject_lora_into_model
from .s4_mil import S4Model
from .coarse import CoarseClassifier
import json
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _preprocessing_answer(answer):
    """
    Preprocess the generated answer:
    - Remove unwanted characters
    - Limit to 2 unique sentences
    - Capitalize properly
    - Ensure clean punctuation
    """
    # Remove unwanted special characters and newlines
    answer = re.sub(r'[{}*^%$#\\]', '', answer).strip()
    # remove first /n
    answer = re.sub(r'^\n+', '', answer).strip()
    # Remove leading ". " or ".\n"
    if answer.startswith(". "):
        answer = answer[2:]
    elif answer.startswith(".\n"):
        answer = answer[2:]

    # Split into sentences
    sentences = re.split(r'\.\s*', answer)

    # Remove empty strings, strip whitespace, and capitalize
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s:
            s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
            if s not in cleaned:
                cleaned.append(s)
            if len(cleaned) == 2:  # Limit to 2 unique sentences
                break

    # Rejoin the cleaned sentences
    answer = '. '.join(cleaned)
    if answer and not answer.endswith('.'):
        answer += '.'

    return answer


def pairwise_aml_loss(embeddings: torch.Tensor,
                         margin: float = 1.0,
                         normalize: bool = False,
                         reduction: str = "mean") -> torch.Tensor:
    """
    Encourages each pair of embeddings to be at least `margin` apart.
    loss_{i,j} = max(0, margin - ||e_i - e_j||_2)

    Args:
        embeddings: [B, D]
        margin: float
        normalize: whether to L2-normalize embeddings before distance
        reduction: 'mean' | 'sum' | 'none'
    Returns:
        scalar loss (or vector for 'none')
    """
    B = embeddings.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=embeddings.requires_grad)

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute pairwise squared distances efficiently
    sq = (embeddings ** 2).sum(dim=1, keepdim=True)  # [B,1]
    dist_sq = sq + sq.t() - 2.0 * (embeddings @ embeddings.t())
    dist_sq = torch.clamp(dist_sq, min=0.0)
    distances = torch.sqrt(dist_sq + 1e-8)  # [B,B]

    # upper triangle indices (i < j)
    idx = torch.triu_indices(B, B, offset=1, device=embeddings.device)
    pair_dists = distances[idx[0], idx[1]]  # [num_pairs]

    # hinge loss per pair
    pair_losses = F.relu(margin - pair_dists)  # [num_pairs]

    if reduction == "mean":
        return pair_losses.mean()
    elif reduction == "sum":
        return pair_losses.sum()
    elif reduction == "none":
        return pair_losses
    else:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")


class AUGUST(torch.nn.Module):
    def __init__(self,
                 llama_hidden_size=4096,
                 fusion_dim=4096,
                 max_answer_length=768,
                 stage="stage_0",
                 latents_dim=1024,

                 ):
        super(AUGUST, self).__init__()

        self.stage = stage
        self.tokenizer = AutoTokenizer.from_pretrained("YBXL/Med-LLaMA3-8B")
        self.llama_model = AutoModelForCausalLM.from_pretrained("YBXL/Med-LLaMA3-8B",
                                                                torch_dtype=torch.bfloat16,
                                                                device_map=None
                                                                )
        self.label_dict = {
            'location': {0: 'antrum', 1: 'body', 2: 'cardia', 3: 'fundus', 4: 'prepylorus'},
            'helicobacter': {0: 'negative', 1: 'positive'},
            'category': {0: 'inflammatory disease', 1: 'benign tumor', 2: 'dysplasia', 3: 'cancer'}
        }
        self.latents_dim = latents_dim
        self._process_bad_tokens()
        # frozen llama model
        for param in self.llama_model.parameters():
            param.requires_grad = False
        self.slide_encoder = S4Model(in_dim=1024, act='gelu', dropout=0.25)
        self.slide_projection = nn.Sequential(
            nn.Linear(512, fusion_dim),
            nn.GELU(),
        )
        self.coarse_classifier = CoarseClassifier(label_dict=self.label_dict, device=device)

        # Multimodal fusion layer
        self.fusion_adapter = nn.Sequential(
            nn.Linear(fusion_dim + llama_hidden_size, llama_hidden_size),
            nn.LayerNorm(llama_hidden_size),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        # llama3 configuration
        self.llama_config = self.llama_model.config
        self.max_answer_length = max_answer_length

        self.question_patch_projector = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.LayerNorm(1024)
        )

    def lora(self):
        for params in self.parameters():
            params.requires_grad = True
        for target in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj","gate_proj", "down_proj"]:
            self.llama_model = inject_lora_into_model(self.llama_model, r=16, target_module_name=target)

    def _process_bad_tokens(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_slide_embedding(self, features, question_text=None):
        """Extract slide features using TITAN encoder"""
        if isinstance(features, str):
            # Convert list of features tensor
            features = torch.load(features)  # N x D

        features = features.to(device)
        question_features = self._get_question_embedding(question_text).unsqueeze(0)  # [1, llama_hidden_size]
        # add question feature to feature tensor
        question_feature_patch = self.question_patch_projector(question_features)
        features = torch.cat([features, question_feature_patch], dim=0)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
            slide_features = self.slide_encoder(features)
        return slide_features.squeeze(), question_features

    def _get_question_embedding(self, question_text):
        """Encode question using LLama3's text understanding"""
        # Tokenize question
        question_inputs = self.tokenizer(
            question_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_answer_length
        ).to(device)

        # Get BioGPT embeddings for the question
        with torch.no_grad():
            question_embeddings = self.llama_model.get_input_embeddings(
            )(question_inputs.input_ids)

        # Average pooling over the sequence length
        question_mask = question_inputs.attention_mask.unsqueeze(-1)
        question_features = (question_embeddings * question_mask).sum(dim=1) / question_mask.sum(dim=1)

        return question_features.squeeze(0)

    def _get_target_dict_from_answer(self, caption):
        """Convert target answer string to label dictionary"""
        target_dict = {}
        for key, label_map in self.label_dict.items():
            for label_id, label_str in label_map.items():
                if label_str in caption.lower():
                    target_dict[key] = label_id
                    break
        return target_dict

    def forward(self, slide_features, question_text, question_features, target_answer=None, caption=None,
                true_label=None,
                task_labels=None):

        visual_features = self.slide_projection(slide_features.unsqueeze(0))  # [1, fusion_dim]
        # Multimodal fusion
        combined_features = torch.cat([visual_features, question_features], dim=-1)
        fused_features = self.fusion_adapter(combined_features)  # [1, llama_hidden_size]
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            if self.training:
                if caption is not None:
                    target_dict = self._get_target_dict_from_answer(caption)
                else:
                    target_dict = self._get_target_dict_from_answer(target_answer)
                if self.stage == "stage_0":
                    return self.coarse_classifier(slide_features, target_dict) \
                           + self.caption_loss(question_text, fused_features, target_answer)

                if self.stage == "stage_1" or self.stage == "stage_2":
                    gen_loss, answer_logits, answer_labels = self.agl_loss(question_text, fused_features,
                                                                                target_answer)
                    token_loss = self.stl_loss(true_label, task_labels, answer_logits, answer_labels)
                    return gen_loss \
                        , self.coarse_classifier(slide_features, target_dict), token_loss
            else:
                return self._inference_forward(question_text, fused_features)

    def caption_loss(self, question_text, fused_features, target_answer):
        """Training with teacher forcing using llama3 - supervise only available parts"""

        # Build structured template (always same order)
        structured_answer = (
            "location: {loc}\n"
            "helicobacter pylori: {hp}\n"
            "condition: {cond}"
        )

        # Extract parts from target_answer
        parts = {"loc": "", "hp": "", "cond": ""}
        for key in ["location", "helicobacter pylori", "condition"]:
            if f"{key}:" in target_answer.lower():
                # Keep original substring
                start = target_answer.lower().index(f"{key}:")
                # Extract until next key or end
                end_candidates = [
                    target_answer.lower().find(k + ":", start + 1)
                    for k in ["location", "helicobacter pylori", "condition"]
                    if target_answer.lower().find(k + ":", start + 1) != -1
                ]
                end = min(end_candidates) if end_candidates else len(target_answer)
                parts_map = {"location": "loc", "helicobacter pylori": "hp", "condition": "cond"}
                parts[parts_map[key]] = target_answer[start:end].split(":", 1)[-1].strip()

        # Fill template
        full_answer = structured_answer.format(
            loc=parts["loc"],
            hp=parts["hp"],
            cond=parts["cond"]
        )

        # Full prompt
        full_prompt = f"{question_text}\n{full_answer}{self.tokenizer.eos_token}"

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_answer_length
        ).to(device)

        # Question-only tokens (for slicing answer part)
        question_inputs = self.tokenizer(
            f"{question_text}\n",
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)
        question_length = question_inputs.input_ids.shape[1]

        outputs = self.get_output_embeddings(inputs, fused_features)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()

        # Slice answer part
        answer_logits = shift_logits[:, question_length - 1:, :]
        answer_labels = shift_labels[:, question_length - 1:]

        # Build label mask: ignore sections with no ground truth
        labels_text = self.tokenizer.decode(answer_labels[0], skip_special_tokens=False)

        ignore_mask = torch.zeros_like(answer_labels)  # 0 = keep, 1 = ignore

        if parts["loc"] == "":
            loc_start = labels_text.find("location:")
            loc_end = labels_text.find("\nhelicobacter pylori:")
            if loc_start != -1:
                ignore_mask[:, loc_start:loc_end] = 1

        if parts["hp"] == "":
            hp_start = labels_text.find("helicobacter pylori:")
            hp_end = labels_text.find("\ncondition:")
            if hp_start != -1:
                ignore_mask[:, hp_start:hp_end] = 1

        if parts["cond"] == "":
            cond_start = labels_text.find("condition:")
            if cond_start != -1:
                ignore_mask[:, cond_start:] = 1

        # Apply mask → set ignored tokens to pad_token_id
        answer_labels = answer_labels.masked_fill(ignore_mask.bool(), self.tokenizer.pad_token_id)

        # Loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(answer_logits.view(-1, answer_logits.size(-1)), answer_labels.view(-1))
        return loss

    def get_output_embeddings(self, inputs, fused_features):
        # --- Embed tokens and inject visual features at first token ---
        input_embeddings = self.llama_model.get_input_embeddings()(inputs.input_ids)
        visual_enhanced_embeddings = input_embeddings.clone()
        visual_enhanced_embeddings[:, 0:1, :] += fused_features  # inject at first token

        # --- Forward pass ---
        outputs = self.llama_model(
            inputs_embeds=visual_enhanced_embeddings,
            attention_mask=inputs.attention_mask,
            labels=inputs.input_ids
        )
        return outputs

    def agl_loss(
            self,
            question_text,
            fused_features,
            target_answer,
    ):
        """Training with teacher forcing using LLaMA3 - inject visual features at first token"""

        # --- Build full input (question + answer) ---
        full_prompt = f"{question_text}\n{target_answer}{self.tokenizer.eos_token}"
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_answer_length
        ).to(device)

        # Tokenize only the question
        question_only = f"{question_text}\n"
        question_inputs = self.tokenizer(
            question_only,
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)
        question_length = question_inputs.input_ids.shape[1]

        outputs = self.get_output_embeddings(inputs, fused_features)

        # --- Shift for causal loss ---
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()

        # --- Answer logits & labels (skip question tokens) ---
        answer_logits = shift_logits[:, question_length - 1:, :]
        answer_labels = shift_labels[:, question_length - 1:]

        loss_fct = nn.CrossEntropyLoss()
        gen_loss = loss_fct(answer_logits.view(-1, answer_logits.size(-1)),
                            answer_labels.view(-1))
        # print(target_answer)
        predicted_answer_ids = torch.argmax(answer_logits, dim=-1)
        predicted_answer = self.tokenizer.decode(predicted_answer_ids[0], skip_special_tokens=True)
        # if predicted_answer.strip() != target_answer.strip():
        #     print("Predicted: ", predicted_answer)
        #     print("Target:    ", target_answer)

        return gen_loss, answer_logits, answer_labels

    def stl_loss(self, true_label=None,
                       task_labels=None, answer_logits=None, answer_labels=None):
        """
        Modified stl_loss:
        Calculates CrossEntropyLoss on the specific spans of text corresponding to true_label,
        BUT without masking the logits to a restricted vocabulary.
        """
        cls_loss = 0.0

        # We still need true_label and answer_logits/labels.
        # task_labels is kept in the signature for compatibility but is not used for masking anymore.
        if true_label is not None and answer_logits is not None and answer_labels is not None:

            # 1. Normalize true_label to list
            if not isinstance(true_label, list):
                true_label = [true_label]

            # 2. Handle specific string splitting logic from original code
            if 'erosion, ulceration' in true_label:
                true_label.remove('erosion, ulceration')
                true_label.append('erosion')
                true_label.append('ulceration')

            for label in true_label:
                label = label.lower()

                # 3. Add spacing (kept from original logic to ensure tokenizer alignment)
                if "chronic inflammation is" not in label:
                    label = " " + label + ""

                    # 4. Encode the target label to find it in the sequence
                label_ids = self.tokenizer.encode(label, add_special_tokens=False)
                label_ids = torch.tensor(label_ids, device=device).unsqueeze(0)  # [1, L_label]

                # 5. Find the start index of this label within the full answer_labels
                start = None
                # Iterate through the sequence to find the matching subsequence
                for i in range(answer_labels.size(1) - label_ids.size(1) + 1):
                    if torch.equal(answer_labels[0, i:i + label_ids.size(1)], label_ids[0]):
                        start = i
                        break

                # 6. If the label span is found, compute loss
                if start is not None:
                    L = label_ids.size(1)

                    # Extract logits and targets for this specific span
                    span_logits = answer_logits[:, start:start + L, :]  # [1, L, Vocab_Size]
                    span_targets = answer_labels[:, start:start + L]  # [1, L]

                    # --- REMOVED: allowed_ids generation and masking logic ---
                    # We no longer filter "mild", "moderate", "marked" or task_labels.
                    # We compare directly against the full vocabulary.

                    # 7. Standard CrossEntropyLoss
                    ce = nn.CrossEntropyLoss()(
                        span_logits.view(-1, span_logits.size(-1)),
                        span_targets.view(-1)
                    )

                    cls_loss += ce

        return cls_loss

    @torch.no_grad()
    def _inference_forward(self, question_text, fused_features):
        """
        Generate answer autoregressive - inject visual features at first token

        Args:
            question_text (str): The input question string.
            fused_features (torch.Tensor): Fused visual-text feature vector [1, hidden_size].
        Returns:
            predicted_answer (str): The generated answer string.
        """

        # Tokenize question only (without answer)
        question_only = f"{question_text}\n"
        question = self.tokenizer(
            question_only,
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)
        question_length = question.input_ids.shape[1]
        question_inputs = self.tokenizer(
            question_only,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_answer_length
        ).to(device)

        # Prepare input_ids starting with question tokens
        input_ids = question_inputs.input_ids  # shape: [1, question_length]

        # Get initial embeddings for input_ids
        input_embeddings = self.llama_model.get_input_embeddings()(input_ids)

        # Inject visual features to the FIRST token (position 0)
        input_embeddings[:, 0:1, :] += fused_features

        generated_ids = input_ids  # will append tokens here

        for _ in range(30):
            outputs = self.llama_model(
                inputs_embeds=input_embeddings,
                attention_mask=torch.ones(generated_ids.shape, device=device),
            )

            # Get logits of the last token
            next_token_logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]

            # Greedy decoding: pick the highest probability token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # shape: [1, 1]

            # If EOS token generated, stop
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            # Append next token to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Recompute embeddings for the entire sequence including newly generated token
            input_embeddings = self.llama_model.get_input_embeddings()(generated_ids)

            # Inject visual features at FIRST token position (keep consistent)
            input_embeddings[:, 0:1, :] += fused_features

        # Decode generated tokens after question tokens (answer only)
        answer_ids = generated_ids[:, question_length:]
        predicted_answer = self.tokenizer.decode(
            answer_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        predicted_answer = _preprocessing_answer(predicted_answer)

        return predicted_answer


    def get_text_embeddings(self, texts: List[str], disable_grad: bool = False) -> torch.Tensor:
        """
        Batch-encode a list of strings and return pooled embeddings [B, D].
        Set disable_grad=False if you want gradients to flow into embeddings for training.
        """
        if len(texts) == 0:
            return torch.empty((0, 0), device=self.device)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_answer_length
        ).to(device)

        if disable_grad:
            with torch.no_grad():
                token_embs = self.llama_model.get_input_embeddings()(inputs.input_ids)
        else:
            token_embs = self.llama_model.get_input_embeddings()(inputs.input_ids)

        mask = inputs.attention_mask.unsqueeze(-1).to(token_embs.dtype)  # [B, L, 1]
        summed = (token_embs * mask).sum(dim=1)  # [B, D]
        denom = mask.sum(dim=1).clamp_min(1e-9)  # [B, 1]
        pooled = summed / denom  # [B, D]
        return pooled

    # --- process all questions in a JSON file and compute aggregated loss ---
    def aml_loss(self,
                    json_path='collected_answers.json',
                    margin: float = 1.0,
                    normalize: bool = False,
                    disable_grad: bool = False,
                    per_question_reduction: str = "mean",
                    aggregate: str = "mean",
                    ) -> Tuple[float, Dict[str, Dict]]:
        """
        Loads JSON file of shape { question_str: [answer1, answer2, ...], ... },
        computes embeddings for each answer list, computes pairwise margin loss per question,
        and aggregates losses across questions.

        Args:
            json_path: input JSON file path
            margin: margin for pairwise hinge
            normalize: whether to normalize embeddings before computing distances
            disable_grad: if True, embeddings computed with no_grad (loss will not backprop)
            per_question_reduction: reduction for per-question pairwise loss ('mean'|'sum'|'none')
            aggregate: how to combine per-question losses -> 'mean' or 'sum'
            save_embeddings_path: optional path to save all embeddings as a .pt file
        Returns:
            total_loss_value (float), details (dict mapping question -> info)
        """

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        details = {}
        per_question_losses = []
        for question, answers in data.items():
            # sanitize: ensure list of strings
            if not isinstance(answers, list) or len(answers) == 0:
                details[question] = {
                    "num_answers": 0,
                    "loss": 0.0,
                    "note": "no answers or invalid format"
                }
                continue

            # compute embeddings (batch)
            embeddings = self.get_text_embeddings(answers, disable_grad=disable_grad)  # [B, D]

            if embeddings.numel() == 0 or embeddings.shape[0] < 2:
                q_loss = torch.tensor(0.0, device=self.device)
                details[question] = {
                    "num_answers": embeddings.shape[0] if embeddings.numel() else 0,
                    "loss": float(q_loss.item()),
                    "note": "not enough answers for pairwise loss"
                }
                per_question_losses.append(q_loss)

            # compute pairwise margin loss
            q_loss = pairwise_aml_loss(embeddings, margin=margin, normalize=normalize,
                                          reduction=per_question_reduction)
            per_question_losses.append(q_loss)
            details[question] = {
                "num_answers": embeddings.shape[0],
                "loss": float(q_loss.item())
            }

        # aggregate per-question losses
        if len(per_question_losses) == 0:
            total_loss = torch.tensor(0.0, device=self.device)
        else:
            stacked = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l, device=self.device) for l in
                                   per_question_losses])
            if aggregate == "mean":
                total_loss = stacked.mean()
            elif aggregate == "sum":
                total_loss = stacked.sum()
            else:
                raise ValueError("aggregate must be 'mean' or 'sum'")
        loss = total_loss / len(per_question_losses) if len(per_question_losses) > 0 else 0.0
        return loss



    def grad_cam_score(self, patch_inputs, question_text, target_answer):

        feats, _ = self.slide_encoder(
            patch_inputs, return_features=True
        )

        feats = feats.to(torch.float32)
        feats.requires_grad_(True)
        feats.retain_grad()  # ✅ critical

        pooled = torch.max(feats, dim=1).values
        visual_features = self.slide_projection(pooled)

        question_features = (
            self._get_question_embedding(question_text)
                .unsqueeze(0)
                .to(patch_inputs.device)
                .to(torch.float32)
        )

        combined = torch.cat([visual_features, question_features], dim=-1)
        fused = self.fusion_adapter(combined)

        full_prompt = f"{question_text}\n{target_answer}{self.tokenizer.eos_token}"
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_answer_length
        ).to(patch_inputs.device)

        question_only = f"{question_text}\n"
        question_inputs = self.tokenizer(
            question_only,
            return_tensors="pt",
            add_special_tokens=False
        ).to(patch_inputs.device)

        question_length = question_inputs.input_ids.shape[1]

        outputs = self.get_output_embeddings(inputs, fused)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()

        answer_logits = shift_logits[:, question_length - 1:, :]
        answer_labels = shift_labels[:, question_length - 1:]

        log_probs = torch.log_softmax(answer_logits, dim=-1)

        token_log_probs = log_probs.gather(
            dim=-1,
            index=answer_labels.unsqueeze(-1)
        ).squeeze(-1)

        answer_score = token_log_probs.mean()

        return answer_score, feats
