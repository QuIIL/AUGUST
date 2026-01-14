from collections import defaultdict, Counter
import json
import random
import h5py
from torch.utils.data import Dataset
import torch


class TestDataset(Dataset):
    def __init__(self, json_path, num_samples=None, seed=42):
        # Load the JSON file
        with open(json_path, 'r') as f:
            self.data = list(json.load(f).values())
        random.seed(seed)
        if num_samples is not None:
            # Sample a subset of the data
            self.data = random.sample(self.data, min(num_samples, len(self.data)))
        else:
            # Use the entire dataset
            self.data = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        feature_path = item["features_path"]
        features = torch.load(feature_path)  # N x D
        return {
            "slide_id": feature_path.split("/")[-1].split(".")[0],
            'features': features,
            'conversations': item['conversations']
        }


def _augment_patches(patch_feats, rare=False, rare_pool=None):
    """
    Apply patch‑level augmentations before slide encoding.

    Args:
        patch_feats (torch.Tensor): [N, D] patch embeddings.
        rare (bool):              whether this sample is from a rare class.
        rare_pool (list of tuples): list of (feat_path, coord_path) for other rare samples.

    Returns:
        patch_feats, coords: augmented versions.
    """
    N, D = patch_feats.shape
    # 1) Patch Dropout
    drop_prob = 0.5 if rare else 0.25
    mask = torch.rand(N, device=patch_feats.device) > drop_prob
    patch_feats = patch_feats * mask.unsqueeze(1)

    # 2) Block Erasure
    erase_chance = 0.5 if rare else 0.3
    if random.random() < erase_chance:
        block_size = random.randint(1, max(1, N // 10))
        start = random.randint(0, N - block_size)
        patch_feats[start: start + block_size] = 0.0

    # 3) Patch Shuffle
    shuffle_chance = 0.4 if rare else 0.2
    if random.random() < shuffle_chance:
        idx = torch.arange(N)
        k = max(1, int(0.2 * N))
        perm = idx[torch.randperm(N)[:k]]
        idx[:k] = perm
        patch_feats = patch_feats[idx]

    # # 4) Patch Mix (only for rare classes)
    mix_chance = 0.5
    if rare and rare_pool and random.random() < mix_chance:
        other_feat_path = random.choice(rare_pool)
        other_feats = torch.load(other_feat_path)
        N1 = patch_feats.size(0)
        N2 = other_feats.size(0)

        # only mix up to the smaller of the two patch counts
        min_N = min(N1, N2)
        if min_N > 0:
            mix_k = min_N // 2
            # choose indices from [0, min_N)
            mix_idxs = torch.randperm(min_N)[:mix_k]
            # perform the mix safely
            patch_feats[mix_idxs] = other_feats[mix_idxs]
    # print(patch_feats)
    return patch_feats


# Define the question order based on your Q_dict keys
Q_ORDER = [
    "location",
    "h_pylori",
    "severity",
    "gastritis_type",
    "gastritis_detail",
    "benign_type",
    "dysplasia_type",
    "cancer_type",
    "cancer_detail"
]

# Keywords to identify each question type
QUESTION_KEYWORDS = {
    "location": ["general location", "proximal vs. distal"],
    "h_pylori": ["Helicobacter pylori", "positive or negative"],
    "severity": ["category of diagnosis", "inflammatory disease, benign tumor, dysplasia, or cancer"],
    "gastritis_type": ["Sydney system", "chronic inflammation", "neutrophilic activity"],
    "gastritis_detail": ["gastritis", "detail", "additional condition"],  # You'll need to define this
    "benign_type": ["benign tumor", "fundic gland polyp", "hyperplastic polyp"],
    "dysplasia_type": ["dysplastic changes", "grade of dysplasia", "tubular adenoma"],
    "cancer_type": ["malignant features", "carcinoma, malignant lymphoma, or NOS"],
    "cancer_detail": ["malignant tumor type", "specify the relevant detail"]
}


def get_question_type(question):
    """Identify question type based on keywords"""
    question_lower = question.lower()

    for q_type, keywords in QUESTION_KEYWORDS.items():
        if any(keyword.lower() in question_lower for keyword in keywords):
            return q_type

    return "unknown"


def get_sort_priority(sample):
    """Get sort priority based on Q_dict order"""
    question_type = get_question_type(sample["question"])

    try:
        return Q_ORDER.index(question_type)
    except ValueError:
        return 999  # Unknown questions go to the end


class SingleQADataset(Dataset):
    def __init__(self, json_path, task='vqa', augment=True, rare_class_threshold=0.3):
        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        self.samples = []
        self.task = task
        self.augment = augment
        self.rare_class_threshold = rare_class_threshold

        # Store question → list of (label, features_path, etc.)
        question_groups = defaultdict(list)

        for entry in raw_data.values():
            features_path = entry["features_path"]
            caption = entry.get("caption", "").strip()

            for conv in entry["conversations"]:
                question = conv["question"].strip()
                if task == 'caption':
                    question_groups[question].append({
                        "features_path": features_path,
                        "caption": caption,
                        "question": question,
                        "answer": conv["answer"],
                        "label": conv["label"],
                    })
                else:
                    question_groups[question].append({
                        "features_path": features_path,
                        "caption": caption,
                        "question": question,
                        "answer": conv["answer"],
                        "label": conv["label"],
                        "task": conv["task"]
                    })

        # Now compute rare classes per question
        self.rare_labels_per_question = {}
        for question, samples in question_groups.items():
            label_counts = Counter()
            for s in samples:
                # Handle list of labels - count each label in the list
                if isinstance(s["label"], list):
                    for label in s["label"]:
                        label_counts[label] += 1
                else:
                    # Handle single label (in case some are not lists)
                    label_counts[s["label"]] += 1

            total = sum(label_counts.values())
            rare_labels = {
                label for label, count in label_counts.items()
                if (count / total) < self.rare_class_threshold
            }
            self.rare_labels_per_question[question] = rare_labels

            # Add all samples to dataset
            self.samples.extend(samples)
        self.rare_pool = defaultdict(list)
        for q, samples in question_groups.items():
            rare_labels = self.rare_labels_per_question[q]
            for s in samples:
                # Check if any label in the list is a rare label
                if isinstance(s["label"], list):
                    # Check if any label in the list intersects with rare_labels
                    if any(label in rare_labels for label in s["label"]):
                        # store the *paths* so you can load on‑the‑fly
                        self.rare_pool[q].append(s["features_path"])
                else:
                    # Handle single label (in case some are not lists)
                    if s["label"] in rare_labels:
                        self.rare_pool[q].append(s["features_path"])

        # Optional: Sample weights (used for WeightedRandomSampler)
        all_labels = [label for s in self.samples for label in s['label']]
        global_label_counts = Counter(all_labels)
        total = sum(global_label_counts.values())
        self.label_weights = {label: total / count for label, count in global_label_counts.items()}

        # For sample weights, you could use average weight of all labels in the sample
        self.sample_weights = [
            sum(self.label_weights[label] for label in s['label']) / len(s['label'])
            for s in self.samples
        ]
        self.samples.sort(key=get_sort_priority)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.load(sample["features_path"])  # N x D
        is_rare = any(
            label in self.rare_labels_per_question.get(sample["question"], set())
            for label in sample["label"]
        )
        if self.augment:
            pool = self.rare_pool.get(sample["question"], [])

            features = _augment_patches(features, rare=is_rare, rare_pool=pool)
        if self.task == 'caption':
            task_labels = 'caption'
        else:
            task_labels = sample["task"]
        return {
            "features": features,
            "caption": sample["caption"],
            "question": sample["question"],
            "answer": sample["answer"],
            "label": sample["label"],
            "task": task_labels,
        }


def get_coarse_fine_question_type(question):
    if "proximal vs. distal stomach" in question:
        return 'location'
    if 'positive or negative' in question:
        return 'helicobacter'
    if 'determine the category of diagnosis' in question:
        return 'category'
    return 'other'


class MultiVQADataset(Dataset):
    """Dataset for multi-turn conversation training (Stage 2) - Simple approach"""

    def __init__(self, json_path, num_samples=None):
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.data = list(data.values())

        if num_samples is not None:
            self.data = random.sample(self.data, min(num_samples, len(self.data)))

        # Expand to conversation-level samples
        self.conversation_samples = []
        for item in self.data:
            self._create_single_turn_samples(item)
            self._create_conversation_samples(item)
        grouped = defaultdict(list)
        for s in self.conversation_samples:
            grouped[s["features_path"]].append(s)

        # Flatten back, grouped by feature_path
        self.conversation_samples = [
            sample for _, group in sorted(grouped.items())
            for sample in group
        ]
        self._build_sample_weights()

    def _create_single_turn_samples(self, item):
        """Create single-turn samples from multi-turn conversations"""
        conversations = item["conversations"]
        features_path = item["features_path"]
        caption = item.get("caption", "")
        for turn in conversations:
            self.conversation_samples.append({
                "features_path": features_path,
                "caption": caption,
                "question": turn["question"],
                "answer": turn["answer"],
                "label": turn.get("label", []),
                "task": turn["task"],
                "turn_idx": None,
                "total_turns": len(conversations),
                "mode": "single"
            })

    def _create_conversation_samples(self, item):
        """Create conversation samples with increasing context"""
        conversations = item["conversations"]
        features_path = item["features_path"]
        caption = item.get("caption", "")

        # Create samples for each conversation turn
        for turn_idx in range(len(conversations)):
            # Include all previous turns as context
            context_turns = conversations[:turn_idx]
            current_turn = conversations[turn_idx]
            if turn_idx == 0:
                continue
            # Build conversation history
            conversation_history = ""
            for prev_turn in context_turns:
                conversation_history += f"{prev_turn['question']}\n{prev_turn['answer']}\n\n"

            # Current question with history
            full_question = conversation_history + f"{current_turn['question']}"

            self.conversation_samples.append({
                "features_path": features_path,
                "caption": caption,
                "question": full_question,
                "answer": current_turn["answer"],
                "label": current_turn.get("label", []),  # ✅ Keep label
                "task": current_turn["task"],
                "turn_idx": turn_idx,
                "total_turns": len(conversations),
                "mode": "multi-turn"
            })

    def _build_sample_weights(self):
        """Compute inverse-frequency label weights (like in single dataset)."""
        all_labels = []
        for s in self.conversation_samples:
            lbls = s.get("label", [])
            if isinstance(lbls, list):
                all_labels.extend(lbls)
            else:
                all_labels.append(lbls)

        # Handle missing labels gracefully
        if len(all_labels) == 0:
            self.sample_weights = [1.0] * len(self.conversation_samples)
            return

        label_counts = Counter(all_labels)
        total = sum(label_counts.values())
        label_weights = {label: total / count for label, count in label_counts.items()}

        # Average weight per sample
        self.sample_weights = []
        for s in self.conversation_samples:
            lbls = s.get("label", [])
            if not lbls:
                self.sample_weights.append(1.0)
            else:
                avg_w = sum(label_weights[l] for l in lbls if l in label_weights) / len(lbls)
                self.sample_weights.append(avg_w)

    def __len__(self):
        return len(self.conversation_samples)

    def __getitem__(self, idx):
        sample = self.conversation_samples[idx]
        features = torch.load(sample["features_path"])  # N x D
        features = _augment_patches(features, rare=False, rare_pool=[])
        return {
            "feature_path": sample["features_path"],
            'features': features,
            'caption': sample['caption'],
            'question': sample['question'],
            'answer': sample['answer'],
            'label': sample.get('label', []),
            'task': sample['task'],
            'turn_idx': sample['turn_idx'],
            'total_turns': sample['total_turns']
        }
