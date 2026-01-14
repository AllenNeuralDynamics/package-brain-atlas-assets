import pandas as pd
from pathlib import Path

metadata_dir = Path('/root/capsule/data/abc_atlas/metadata/Allen-CCF-2020/20230630/')
pt_df = pd.read_csv(metadata_dir / "parcellation_term.csv")
pptm_df = pd.read_csv(metadata_dir / "parcellation_to_parcellation_term_membership.csv")
parcellation_df = pd.read_csv(metadata_dir / "parcellation.csv")
ptsm_df = pd.read_csv(metadata_dir / "parcellation_term_set_membership.csv")

def _to_annotation_label(identifier):
	if isinstance(identifier, str):
		suffix = identifier.split(':', 1)[-1]
		# parcellation.csv label convention includes the year segment
		return f"AllenCCF-Annotation-2020-{suffix}"
	return pd.NA


pt_df['annotation_label'] = pt_df['identifier'].apply(_to_annotation_label)


# Pull `parcellation_index` onto pt_df (matches are via the derived annotation_label)
pt_df = pt_df.merge(
	parcellation_df[['label', 'parcellation_index']].rename(columns={'label': 'annotation_label'}),
	on='annotation_label',
	how='left',
)

# Ensure `parcellation_index` is integer, minting new ids for rows that didn't map.
pt_df['parcellation_index'] = pd.to_numeric(pt_df['parcellation_index'], errors='coerce')

missing_mask = pt_df['parcellation_index'].isna()
if missing_mask.any():
	# Continue counting from the current max parcellation_index in parcellation.csv.
	# (If parcellation_df is empty for some reason, start at 1.)
	max_existing = pd.to_numeric(parcellation_df['parcellation_index'], errors='coerce').max()
	max_existing = int(max_existing) if pd.notna(max_existing) else 0
	new_ids = range(max_existing + 1, max_existing + 1 + int(missing_mask.sum()))
	pt_df.loc[missing_mask, 'parcellation_index'] = list(new_ids)

# Now we can safely cast to a true integer dtype.
pt_df['parcellation_index'] = pt_df['parcellation_index'].astype(int)

# Pull `parcellation_term_set_label` onto pt_df (matches are via parcellation_term.label)
pt_df = pt_df.merge(
	ptsm_df[['parcellation_term_label', 'parcellation_term_set_label']].drop_duplicates(),
	how='left',
	left_on='label',
	right_on='parcellation_term_label',
)

# Keep only the new value column (avoid retaining a duplicated join key)
pt_df = pt_df.drop(columns=['parcellation_term_label'])


def _dedupe_by_parcellation_index(df: pd.DataFrame) -> pd.DataFrame:
	"""Deduplicate rows that share the same parcellation_index.

	Rules:
	1) Keep the row whose `label` starts with "AllenCCF" (if present).
	2) Replace `parcellation_term_set_label` with a list of all unique values
	   across the duplicate rows.

	Notes:
	- Rows with missing parcellation_index are left as-is.
	- If multiple rows match rule (1), the first encountered is kept.
	"""
	if df.empty:
		return df

	# Ensure we can represent merged label sets.
	if 'parcellation_term_set_label' not in df.columns:
		return df

	# Work row-wise on groups with non-null parcellation_index.
	with_index = df.dropna(subset=['parcellation_index']).copy()
	without_index = df[df['parcellation_index'].isna()].copy()

	def _merge_group(g: pd.DataFrame) -> pd.Series:
		# Collect unique, non-null term set labels.
		labels = (
			g['parcellation_term_set_label']
			.dropna()
			.astype(str)
			.unique()
			.tolist()
		)
		labels_sorted = sorted(labels)

		# Choose the canonical row.
		mask = g['label'].astype(str).str.startswith('AllenCCF', na=False)
		if mask.any():
			row = g.loc[mask].iloc[0].copy()
		else:
			row = g.iloc[0].copy()

		row['parcellation_term_set_label'] = labels_sorted
		return row

	rows = []
	for _, g in with_index.groupby('parcellation_index', sort=False, dropna=False):
		rows.append(_merge_group(g))
	# If there were no non-null parcellation_index rows, keep an empty frame.
	deduped = pd.DataFrame(rows) if rows else with_index.iloc[0:0].copy()

	# Preserve original-ish order: keep the non-null-index deduped rows first,
	# then any rows that never had an index.
	return pd.concat([deduped, without_index], ignore_index=True)


pt_df = _dedupe_by_parcellation_index(pt_df)

# Quick validation: coverage + uniqueness of parcellation_index mapping
_mapped = pt_df.dropna(subset=['parcellation_index'])
_missing_indices = set(parcellation_df['parcellation_index']) - set(
	_mapped['parcellation_index'].astype(int)
)
_dup_counts = _mapped.groupby('parcellation_index').size()
_nonunique = _dup_counts[_dup_counts > 1]

print(f"disinct parcellation_index in parcellation.csv: {parcellation_df['parcellation_index'].nunique()}")
print(f"distinct parcellation_index mapped in pt_df: {_mapped['parcellation_index'].nunique()}")
print(f"missing parcellation_index: count {len(_missing_indices)}, {_missing_indices}")
print(f"non-unique parcellation_index count: {len(_nonunique)}")

if len(_missing_indices):
	print("example missing parcellation_index values:", sorted(list(_missing_indices))[:20])

if len(_nonunique):
	print("top repeated parcellation_index values (index: count):")
	print(_nonunique.sort_values(ascending=False).head(20).to_string())

print(pt_df.head())

pt_df.to_csv("/scratch/parcellation_term_enriched.csv", index=False)