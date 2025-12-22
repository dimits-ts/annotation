import itertools
import urllib
import time

import pandas as pd
import numpy as np

WHOW_URLS = [
    "https://drive.google.com/file/d/1pIbhn_HhHOEAODweW0jYNhZaxyXc5muT",
    "https://docs.google.com/spreadsheets/d/1GTAmX9RxQF7PyKy0asGM1ShjtRbrOUTB",
    "https://docs.google.com/spreadsheets/d/13siGnX-hd3VWSkCOzNs-enVckE30_Yfq",
    "https://docs.google.com/spreadsheets/d/1iX-CwUXMzvqiWvV0ZY8T86uyESRmhNEzuHVTak4CCUQ",
]

FORA_URLS = [
    "https://drive.google.com/file/d/1R59SvIhWe1rkvKRyeAtebyqJ4dcYNORB",
    "https://docs.google.com/spreadsheets/d/1uhRvODVCU1N19Vz8rp-y7XFMWhupPlA_",
    "https://docs.google.com/spreadsheets/d/1A699mJ2C5zGBYiH7ATbonq7uZF2lJx0O",
    "https://docs.google.com/spreadsheets/d/1Zc65DZ8PINabwcaaadbDLCx9VF6ecIpEDIaiHl-solQ",
]

PARK_URLS = [
    "https://docs.google.com/spreadsheets/d/1siHft7uMPi6pfg6GdITFl-6YUfhTwSOaRIvoEXmuYQQ",
    "https://docs.google.com/spreadsheets/d/1X3ATMStj8cBd6iZ9KZF6cdahhiDbkpJMWu8q5P6tq04",
    "https://docs.google.com/spreadsheets/d/1OIDv86wSsaTJkH37QYB60_TwE_DD_7boh9-_pztusTU",
    "https://docs.google.com/spreadsheets/d/1C-E_TYiHLlKRrC3hmcAYL3eOlhkA8HOPF0SkEP79j2Y",
]


def analyze(urls: list[str]) -> None:
    ann_cols = []
    for url in urls:
        try:
            df = open_sheet(url)
            if df is not None:
                ann_cols.append(df.Categories)
        except Exception as e:
            print(f"Error when processing url {url}: {e}")

    print(
        "Mean pairwise agreement: " f"{dice_annotations(ann_cols) * 100:.3f}%"
    )

    print("Binary agreement on `None` category:")
    agreements = binary_none_agreement(ann_cols)
    for (i, j), score in agreements.items():
        print(f"Annotator {i} vs {j}: {score*100:.3f}%")

    print("Categories associated with `None`:")
    df = category_none_association(ann_cols)

    print(df.head(10))


def dice_coefficient(list1: str, list2: str) -> float:
    set1 = set(x.strip() for x in list1.split(","))
    set2 = set(x.strip() for x in list2.split(","))

    if not set1 and not set2:
        return 1.0

    return 2 * len(set1 & set2) / (len(set1) + len(set2))


def parse_labels(cell: str) -> set[str]:
    if cell in {"None", "", np.nan}:
        return set()
    return set(x.strip() for x in cell.split(","))


def category_none_association(
    annotations: list[pd.Series],
) -> pd.DataFrame:
    """
    For each category, compute how strongly it is associated
    with other annotators assigning None.
    """
    categories = set()
    for ann in annotations:
        for cell in ann.dropna():
            categories |= parse_labels(cell)

    records = []

    n_ann = len(annotations)

    for cat in sorted(categories):
        none_when_cat = []

        for i in range(n_ann):
            for j in range(n_ann):
                if i == j:
                    continue

                a_i = annotations[i].fillna("None")
                a_j = annotations[j].fillna("None")

                for x_i, x_j in zip(a_i, a_j):
                    has_cat = cat in parse_labels(x_i)
                    is_none_j = x_j == "None"

                    if has_cat:
                        none_when_cat.append(is_none_j)

        if none_when_cat:
            p_none_given_cat = np.mean(none_when_cat)

            records.append(
                {
                    "category": cat,
                    "P(None | cat)": p_none_given_cat,
                    "support": len(none_when_cat),
                }
            )

    return pd.DataFrame(records).sort_values("P(None | cat)", ascending=False)


def binary_none_agreement(
    annotations: list[pd.Series],
) -> dict[tuple[int, int], float]:
    """
    Computes pairwise binary agreement on whether an annotation is None.

    Returns:
        dict mapping (i, j) -> agreement score
    """
    agreements = {}

    for (i, ann1), (j, ann2) in itertools.combinations(
        enumerate(annotations), 2
    ):
        s1 = ann1.fillna(-1)
        s2 = ann2.fillna(-1)

        agreement = ((s1 == -1) == (s2 == -1)).mean()
        agreements[(i, j)] = agreement

    return agreements


def dice_annotations(all_annotations: list[pd.Series]) -> float:
    iaa = []

    for ann1, ann2 in itertools.combinations(all_annotations, 2):
        pairwise_mean = (
            pd.DataFrame({"a1": ann1, "a2": ann2})
            .fillna("None")
            .apply(lambda x: dice_coefficient(x["a1"], x["a2"]), axis=1)
            .mean()
        )
        iaa.append(pairwise_mean)

    return np.mean(iaa)


def open_sheet(url: str) -> pd.DataFrame:
    # changes to sheets can be live patched
    ts = int(time.time())
    cache_bust_url = f"&cache_bust={ts}"
    try:
        # Try native Google Sheets export
        return pd.read_csv(f"{url}/export?format=csv{cache_bust_url}")
    except urllib.error.HTTPError:
        # Fallback: assume Drive-hosted ODS
        file_id = url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}{cache_bust_url}"
        return pd.read_excel(download_url, engine="odf")


def main():
    datasets = {"WHoW": WHOW_URLS, "Fora": FORA_URLS, "Park": PARK_URLS}

    for ds_name, ds_urls in datasets.items():
        print(f"=================== {ds_name} ===================")
        analyze(urls=ds_urls)


if __name__ == "__main__":
    main()
