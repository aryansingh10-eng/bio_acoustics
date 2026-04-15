"""
migration_data.py
Provides migration / phenology data for common bird species.

Data is based on eBird frequency curves (North America defaults).
Each species entry has:
    - peak_months: list of months (1-12) when the species is most commonly detected
    - presence: dict mapping month → relative frequency (0.0 – 1.0)
    - migration_type: 'resident' | 'short' | 'neotropical' | 'irruptive'
    - notes: short human-readable text
    - wintering_range: rough description
    - breeding_range: rough description

To add more species: follow the same schema and add to SPECIES_PHENOLOGY.

For production use, replace this with a live eBird API call:
    GET https://api.ebird.org/v2/product/lists/{regionCode}
"""

from __future__ import annotations

# ── Phenology database ────────────────────────────────────────────────────────
# Monthly presence frequency (0=absent, 1.0=peak)
# Months: Jan=1 … Dec=12

SPECIES_PHENOLOGY: dict[str, dict] = {

    # ── Neotropical migrants ─────────────────────────────────────────────────
    "american_redstart": {
        "common_name":     "American Redstart",
        "peak_months":     [5, 6, 8, 9],
        "presence": {1:0, 2:0, 3:0.05, 4:0.3, 5:0.9, 6:1.0, 7:0.8, 8:0.9, 9:0.7, 10:0.2, 11:0.02, 12:0},
        "migration_type":  "neotropical",
        "spring_arrival":  "late April – early May",
        "fall_departure":  "September – October",
        "wintering_range": "Caribbean, Central America, northern South America",
        "breeding_range":  "Eastern North America, southern Canada",
        "notes":           "One of the most common warblers during spring migration.",
    },
    "blackpoll_warbler": {
        "common_name":     "Blackpoll Warbler",
        "peak_months":     [5, 9],
        "presence": {1:0, 2:0, 3:0, 4:0.1, 5:0.95, 6:0.6, 7:0.4, 8:0.5, 9:0.9, 10:0.3, 11:0, 12:0},
        "migration_type":  "neotropical",
        "spring_arrival":  "May",
        "fall_departure":  "September – October (transoceanic flight)",
        "wintering_range": "Northern South America",
        "breeding_range":  "Boreal forest, Alaska to Newfoundland",
        "notes":           "Undertakes one of the longest overwater migrations of any songbird.",
    },
    "wood_thrush": {
        "common_name":     "Wood Thrush",
        "peak_months":     [5, 6, 7, 8],
        "presence": {1:0, 2:0, 3:0, 4:0.2, 5:0.85, 6:1.0, 7:1.0, 8:0.9, 9:0.5, 10:0.1, 11:0, 12:0},
        "migration_type":  "neotropical",
        "spring_arrival":  "late April",
        "fall_departure":  "September – October",
        "wintering_range": "Central America",
        "breeding_range":  "Eastern deciduous forest",
        "notes":           "Population declined ~60% since 1966 due to habitat loss.",
    },

    # ── Short-distance migrants ───────────────────────────────────────────────
    "american_robin": {
        "common_name":     "American Robin",
        "peak_months":     [3, 4, 5, 6, 7, 8, 9],
        "presence": {1:0.3, 2:0.4, 3:0.85, 4:1.0, 5:1.0, 6:1.0, 7:1.0, 8:1.0, 9:0.9, 10:0.7, 11:0.4, 12:0.3},
        "migration_type":  "short",
        "spring_arrival":  "February – March",
        "fall_departure":  "October – November (partial)",
        "wintering_range": "Southern US, Mexico",
        "breeding_range":  "Most of North America",
        "notes":           "Many populations resident year-round in milder climates.",
    },
    "dark_eyed_junco": {
        "common_name":     "Dark-eyed Junco",
        "peak_months":     [11, 12, 1, 2, 3],
        "presence": {1:1.0, 2:1.0, 3:0.9, 4:0.5, 5:0.2, 6:0.1, 7:0.1, 8:0.15, 9:0.3, 10:0.7, 11:1.0, 12:1.0},
        "migration_type":  "short",
        "spring_arrival":  "March – April (departs lowlands)",
        "fall_departure":  "September – October (arrives lowlands)",
        "wintering_range": "Southern US, Mexico",
        "breeding_range":  "Boreal forest, mountain ranges",
        "notes":           "Classic 'snowbird' — appears at feeders in winter.",
    },

    # ── Residents ────────────────────────────────────────────────────────────
    "northern_cardinal": {
        "common_name":     "Northern Cardinal",
        "peak_months":     list(range(1, 13)),
        "presence": {m: 1.0 for m in range(1, 13)},
        "migration_type":  "resident",
        "spring_arrival":  "N/A (year-round)",
        "fall_departure":  "N/A",
        "wintering_range": "Eastern US, Mexico",
        "breeding_range":  "Same as wintering",
        "notes":           "Non-migratory. Present year-round throughout range.",
    },
    "house_sparrow": {
        "common_name":     "House Sparrow",
        "peak_months":     list(range(1, 13)),
        "presence": {m: 1.0 for m in range(1, 13)},
        "migration_type":  "resident",
        "spring_arrival":  "N/A",
        "fall_departure":  "N/A",
        "wintering_range": "Global (introduced)",
        "breeding_range":  "Global (introduced)",
        "notes":           "Introduced species, abundant year-round near human settlements.",
    },

    # ── Indian subcontinent species (for Bengaluru context) ──────────────────
    "asian_koel": {
        "common_name":     "Asian Koel",
        "peak_months":     [3, 4, 5, 6, 7],
        "presence": {1:0.3, 2:0.4, 3:0.9, 4:1.0, 5:1.0, 6:0.9, 7:0.8, 8:0.5, 9:0.4, 10:0.3, 11:0.3, 12:0.3},
        "migration_type":  "short",
        "spring_arrival":  "February – March (breeding season onset)",
        "fall_departure":  "August – September",
        "wintering_range": "Southern India, Sri Lanka",
        "breeding_range":  "South and Southeast Asia",
        "notes":           "Brood parasite of House Crow; loud call announces monsoon.",
    },
    "barn_swallow": {
        "common_name":     "Barn Swallow",
        "peak_months":     [3, 4, 5, 9, 10],
        "presence": {1:0.1, 2:0.2, 3:0.8, 4:1.0, 5:1.0, 6:0.9, 7:0.8, 8:0.7, 9:0.9, 10:0.8, 11:0.4, 12:0.1},
        "migration_type":  "neotropical",
        "spring_arrival":  "March",
        "fall_departure":  "October – November",
        "wintering_range": "Sub-Saharan Africa, South Asia",
        "breeding_range":  "Europe, North America, temperate Asia",
        "notes":           "Among the most widespread migrants; travels up to 10,000 km.",
    },
    "common_swift": {
        "common_name":     "Common Swift",
        "peak_months":     [4, 5, 6, 7, 8],
        "presence": {1:0, 2:0, 3:0.1, 4:0.8, 5:1.0, 6:1.0, 7:1.0, 8:0.9, 9:0.5, 10:0.1, 11:0, 12:0},
        "migration_type":  "neotropical",
        "spring_arrival":  "April",
        "fall_departure":  "August – September",
        "wintering_range": "Sub-Saharan Africa",
        "breeding_range":  "Europe, western Asia",
        "notes":           "Spends almost its entire life airborne; only lands to breed.",
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _normalize_key(species: str) -> str:
    """Map raw classifier label to a phenology key."""
    return species.lower().replace(" ", "_").replace("-", "_")


def get_phenology(species_label: str) -> dict | None:
    """
    Returns phenology dict for *species_label* or None if unknown.
    Tries exact match first, then partial match on common_name.
    """
    key = _normalize_key(species_label)
    if key in SPECIES_PHENOLOGY:
        return SPECIES_PHENOLOGY[key]

    # Fuzzy: match on common_name
    label_lower = species_label.lower()
    for data in SPECIES_PHENOLOGY.values():
        if label_lower in data["common_name"].lower() or data["common_name"].lower() in label_lower:
            return data

    return None


def seasonal_confidence_adjustment(
    species_label: str,
    month: int,
    raw_confidence: float,
) -> tuple[float, str]:
    """
    Adjusts the classifier's raw confidence by seasonal presence probability.

    Returns:
        (adjusted_confidence, explanation_string)
    """
    pheno = get_phenology(species_label)
    if pheno is None:
        return raw_confidence, "No migration data available for this species."

    presence = pheno["presence"].get(month, 0.0)

    # Bayesian-style blend: w=0.3 weight to phenology prior
    w_pheno = 0.3
    adjusted = raw_confidence * (1 - w_pheno) + raw_confidence * presence * w_pheno
    adjusted = min(adjusted, 1.0)

    m_name = MONTH_NAMES[month]
    if presence >= 0.7:
        context = f"Expected season ({m_name} presence: {presence:.0%}). Confidence boosted."
    elif presence >= 0.3:
        context = f"Transitional period ({m_name} presence: {presence:.0%})."
    elif presence > 0:
        context = f"Uncommon this month ({m_name} presence: {presence:.0%}). Confidence reduced."
    else:
        context = f"Not expected in {m_name} (presence: 0%). Possible out-of-range sighting."

    return adjusted, context


def migration_calendar_text(species_label: str) -> str:
    """Returns a formatted multi-line migration summary string."""
    pheno = get_phenology(species_label)
    if pheno is None:
        return f"No migration data found for '{species_label}'."

    peak = ", ".join(MONTH_NAMES[m] for m in pheno["peak_months"])
    lines = [
        f"**{pheno['common_name']}**",
        f"Migration type: {pheno['migration_type'].capitalize()}",
        f"Peak months: {peak}",
        f"Spring arrival: {pheno['spring_arrival']}",
        f"Fall departure: {pheno['fall_departure']}",
        f"Breeding range: {pheno['breeding_range']}",
        f"Wintering range: {pheno['wintering_range']}",
        f"Notes: {pheno['notes']}",
    ]
    return "\n".join(lines)
