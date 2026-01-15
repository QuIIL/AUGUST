from torch import nn
import torch

class CoarseClassifier(nn.Module):
    def __init__(self, label_dict=None, device=None):
        super(CoarseClassifier, self).__init__()
        self.adapter_coarse_branch = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.2)
        )
        self.device = device
        # coarse-classifier
        self.location_classifier = nn.Linear(1024, 5)  # 5 ['antrum', 'body', 'cardia', 'fundus', 'prepylorus']
        self.helicobacter_classifier = nn.Linear(1024, 2)  # ['negative', 'positive']
        self.category_classifier = nn.Linear(1024, 4)
        self.label_dict = label_dict

    def forward(self, slide_features, target_dict):
        loss = 0.0
        tasks = {
            "location": self.location_classifier,
            "helicobacter": self.helicobacter_classifier,
            "category": self.category_classifier,
        }
        slide_features = self.adapter_coarse_branch(slide_features)
        for key, classifier in tasks.items():
            if key in target_dict and target_dict[key] is not None:
                labels = torch.tensor(target_dict[key], device=self.device)
                logits = classifier(slide_features)
                loss += F.cross_entropy(logits, labels)
        return loss
