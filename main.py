import os
import pickle
from typing import List, Dict, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class GeneTokenizer:
	"""
	Tokenizer for gene names; adds special tokens including [CLS].
	"""

	def __init__(self, gene_names: List[str]):
		self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}
		self.gene_to_id: Dict[str, int] = {
			gene: idx + len(self.special_tokens) for idx, gene in enumerate(gene_names)
		}
		self.id_to_gene: Dict[int, str] = {v: k for k, v in self.gene_to_id.items()}

	def encode_genes(self, genes: List[str]) -> List[int]:
		return [self.gene_to_id.get(g, self.special_tokens["[UNK]"]) for g in genes]

	@property
	def vocab_size(self) -> int:
		return len(self.special_tokens) + len(self.gene_to_id)


class GeneSequenceDataset(Dataset):
	"""
	Builds gene token sequences per row: genes sorted by descending expression, prefixed by [CLS].
	"""

	def __init__(self, csv_path: str, tokenizer: GeneTokenizer, max_seq_len: int = 1024):
		df = pd.read_csv(csv_path)
		if "identifier" not in df.columns:
			raise ValueError("CSV must contain an 'identifier' column")
		self.gene_cols: List[str] = [c for c in df.columns if c != "identifier"]
		self.df = df
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		row = self.df.iloc[idx]
		expressions = row[self.gene_cols].astype(float)
		sorted_genes = list(expressions.sort_values(ascending=False).index)
		token_ids = self.tokenizer.encode_genes(sorted_genes[: self.max_seq_len])
		token_ids = [self.tokenizer.special_tokens["[CLS]"]] + token_ids
		attention_mask = [1] * len(token_ids)
		return {
			"input_ids": torch.tensor(token_ids, dtype=torch.long),
			"attention_mask": torch.tensor(attention_mask, dtype=torch.long),
		}


def pad_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
	max_len = max(item["input_ids"].shape[0] for item in batch)
	input_ids_padded = []
	attention_mask_padded = []
	for item in batch:
		ids = item["input_ids"]
		mask = item["attention_mask"]
		pad_len = max_len - ids.shape[0]
		if pad_len > 0:
			ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
			mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
		input_ids_padded.append(ids)
		attention_mask_padded.append(mask)
	return {
		"input_ids": torch.stack(input_ids_padded, dim=0),
		"attention_mask": torch.stack(attention_mask_padded, dim=0),
	}


class LargeGeneTransformer(nn.Module):
	"""
	Larger configurable transformer encoder for gene sequences with CLS-based classification.
	"""

	def __init__(
		self,
		vocab_size: int,
		d_model: int = 768,
		nhead: int = 12,
		num_layers: int = 12,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
		num_classes: int | None = None,
	):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, d_model)
		self.pos_emb = nn.Embedding(8192, d_model)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.class_head = nn.Linear(d_model, num_classes) if num_classes is not None else None

	def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
		B, T = input_ids.shape
		positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
		x = self.token_emb(input_ids) + self.pos_emb(positions)
		key_padding_mask = attention_mask == 0
		x = self.transformer(x, src_key_padding_mask=key_padding_mask)
		if self.class_head is None:
			raise ValueError("class_head not defined; provide num_classes to the model")
		cls_repr = x[:, 0, :]
		class_logits = self.class_head(cls_repr)
		return class_logits


def build_tokenizer_from_csv(csv_path: str) -> GeneTokenizer:
	df = pd.read_csv(csv_path, nrows=1)
	if "identifier" not in df.columns:
		raise ValueError("CSV must contain an 'identifier' column")
	gene_cols = [c for c in df.columns if c != "identifier"]
	return GeneTokenizer(gene_cols)


def predict_all(csv_path: str, label_map_path: str) -> Dict[str, int]:
	tokenizer = build_tokenizer_from_csv(csv_path)
	dataset = GeneSequenceDataset(csv_path, tokenizer, max_seq_len=1024)
	loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=pad_batch)

	with open(label_map_path, "rb") as f:
		label_mapping = pickle.load(f)
	num_classes = len(label_mapping)

	# Allow overrides
	d_model = int(os.environ.get("GENE_D_MODEL", "768"))
	nhead = int(os.environ.get("GENE_NHEAD", "12"))
	num_layers = int(os.environ.get("GENE_LAYERS", "12"))
	ff = int(os.environ.get("GENE_FF", "2048"))
	model = LargeGeneTransformer(
		vocab_size=tokenizer.vocab_size,
		d_model=d_model,
		nhead=nhead,
		num_layers=num_layers,
		dim_feedforward=ff,
		num_classes=num_classes,
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	counts: Dict[str, int] = {}
	with torch.no_grad():
		for batch in loader:
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			class_logits = model(input_ids, attention_mask)
			preds = class_logits.argmax(dim=1).cpu().tolist()
			for p in preds:
				label = label_mapping[int(p)]
				counts[label] = counts.get(label, 0) + 1

	return counts


if __name__ == "__main__":
	default_csv = os.path.join(
		os.path.dirname(__file__),
		"data",
		"preprocessed",
		"files_preprocessed.csv",
	)
	csv_path = os.environ.get("GENE_CSV", default_csv)

	default_map = os.path.join(
		os.path.dirname(__file__),
		"data",
		"",
		"5_cohorts_labels_mapping.pkl",
	)
	label_map_path = os.environ.get("COHORTS_LABELS_MAP", default_map)

	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found at {csv_path}. Set GENE_CSV env var.")
	if not os.path.exists(label_map_path):
		raise FileNotFoundError(
			f"Label mapping not found at {label_map_path}. Set COHORTS_LABELS_MAP env var."
		)

	counts = predict_all(csv_path, label_map_path)
	print("Cancer type counts:")
	for label, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
		print(f"{label}: {cnt}")
