"""
Reads a listing's kind from its words: rent or sale, available or not, the type of
property, and the postcode in its locality. Pure text in, labels out.

English, Dutch and French keywords, so Belgian, Dutch and French portals work too.
A site can always name these itself (a `deal` / `availability` / `type` field, or
`deal` / `property_type` in sites.json); detection only fills what is missing.
"""
import re

RENT, SALE = "rent", "sale"
AVAILABLE, UNDER_OFFER, SOLD, RENTED = "available", "under offer", "sold", "rented"


def _rx(*words: str) -> re.Pattern:
    # Whole words only; a pattern that starts with a symbol ("/month") may follow a digit ("€950/month")
    return re.compile(r"(?:(?<![\w-])|(?=[^\w\s]))(?:" + "|".join(words) + r")(?![\w])", re.IGNORECASE)


# URL path / query words, e.g. /for-rent/, /te-koop/, ?transaction=sale
# Whole words of the path or query ("current" and "wholesale" don't count; nor does "location=",
# a search parameter on many sites, though it means "rental" in French paths)
_URL_RENT = re.compile(r"(?<![a-z])(?:for-rent|to-rent|te-huur|a-louer|à-louer|huren|rent|rentals?|lettings?|huur)"
                       r"(?![a-z])", re.IGNORECASE)
_URL_SALE = re.compile(r"(?<![a-z])(?:for-sale|te-koop|a-vendre|à-vendre|vente|kopen|buy|sale|koop)(?![a-z])",
                       re.IGNORECASE)

_TEXT_RENT = _rx(r"for rent", r"to rent", r"to let", r"te huur", r"à louer", r"a louer",
                 r"per month", r"/\s?month", r"/\s?mo", r"p\.?c\.?m", r"per maand", r"/\s?maand", r"par mois",
                 r"/\s?mois", r"monthly rent", r"huurprijs", r"loyer")
_TEXT_SALE = _rx(r"for sale", r"te koop", r"à vendre", r"a vendre", r"asking price", r"vraagprijs",
                 r"prix de vente", r"kosten koper", r"k\.k\.", r"v\.o\.n\.")

# Most specific first: "sold STC" is under offer, not sold
_STATUS = [
    (UNDER_OFFER, _rx(r"under offer", r"under option", r"sold stc", r"sold subject to contract", r"let agreed",
                      r"reserved", r"onder bod", r"onder optie", r"in optie", r"gereserveerd",
                      r"sous option", r"sous compromis", r"compromis signé", r"réservé")),
    (SOLD, _rx(r"sold", r"verkocht", r"vendu", r"vendue")),
    (RENTED, _rx(r"rented", r"let", r"verhuurd", r"loué", r"louée")),
]
# "let" alone is too common in prose; it only counts in a short status text (a badge)
_STATUS_ANYWHERE = [
    (UNDER_OFFER, _STATUS[0][1]),
    (SOLD, _rx(r"sold", r"verkocht", r"vendu", r"vendue")),
    (RENTED, _rx(r"rented out", r"already rented", r"verhuurd", r"loué", r"louée")),
]

# Checked in order: "studio apartment" is a studio, "apartment building" is a building
_TYPES = [
    ("Room", _rx(r"room for rent", r"shared room", r"kamer", r"chambre", r"kot", r"student room")),
    ("Studio", _rx(r"studio", r"studette")),
    ("Penthouse", _rx(r"penthouse")),
    ("Duplex", _rx(r"duplex", r"triplex")),
    ("Loft", _rx(r"loft")),
    ("Apartment building", _rx(r"apartment block", r"apartment building", r"opbrengsteigendom",
                               r"immeuble de rapport", r"mixed-use building")),
    ("Apartment", _rx(r"apartment", r"apartments", r"flat", r"appartement", r"appartment", r"ground floor",
                      r"gelijkvloers", r"rez-de-chaussée")),
    ("Villa", _rx(r"villa")),
    ("Bungalow", _rx(r"bungalow")),
    ("House", _rx(r"house", r"woning", r"huis", r"maison", r"town house", r"townhouse",
                  r"terraced", r"semi-detached", r"detached", r"rijwoning", r"halfopen", r"open bebouwing",
                  r"mansion", r"farmhouse", r"cottage", r"hoeve", r"fermette")),
    ("Land", _rx(r"land", r"plot", r"building plot", r"bouwgrond", r"grond", r"terrain")),
    ("Garage", _rx(r"garage", r"parking", r"parking space", r"staanplaats", r"garage box")),
    ("Office", _rx(r"office", r"kantoor", r"bureau", r"bureaux")),
    ("Commercial", _rx(r"commercial", r"shop", r"retail", r"handelspand", r"winkel", r"commerce",
                       r"warehouse", r"magazijn", r"entrepôt", r"industrial")),
]

_POSTCODE = re.compile(r"^\s*(\d{4}(?:\s?[A-Z]{2})?|\d{5})\b[\s,-]*(.*)$")      # 1050 / 1012 AB / 75011
_POSTCODE_END = re.compile(r"^(.*?)[\s,(-]+(\d{4,5})\)?\s*$")                   # Ixelles (1050)


def detect_deal(url: str = "", text: str = "") -> str | None:
    """'rent', 'sale' or None. The page URL decides first, then the card's words."""
    path = url.split("://", 1)[-1]
    path = path.split("/", 1)[1] if "/" in path else ""
    rent, sale = bool(_URL_RENT.search(path)), bool(_URL_SALE.search(path))
    if rent != sale:
        return RENT if rent else SALE
    rent, sale = bool(_TEXT_RENT.search(text)), bool(_TEXT_SALE.search(text))
    if rent != sale:
        return RENT if rent else SALE
    return None


_LABEL_RENT = re.compile(r"rent|let|huur|louer|location|miete", re.IGNORECASE)
_LABEL_SALE = re.compile(r"sale|sell|buy|koop|vendre|vente|kauf", re.IGNORECASE)


def deal_from_label(label: str) -> str | None:
    """A site's own deal label ('For rent', 'Te koop', 'Sale') → 'rent' / 'sale' / None."""
    rent, sale = bool(_LABEL_RENT.search(label or "")), bool(_LABEL_SALE.search(label or ""))
    return (RENT if rent else SALE) if rent != sale else None


def detect_availability(status_text: str = "", text: str = "") -> str:
    """
    'available', 'under offer', 'sold' or 'rented'. status_text is a badge the site
    marks as the status (read strictly); text is the whole card (read cautiously).
    A listing on a results page with no such mark is available.
    """
    if status_text:
        for label, pattern in _STATUS:
            if pattern.search(status_text):
                return label
    for label, pattern in _STATUS_ANYWHERE:
        if pattern.search(text):
            return label
    return AVAILABLE


def detect_type(text: str) -> str | None:
    """The first property type the text names, e.g. 'Apartment', 'House', 'Studio'."""
    for label, pattern in _TYPES:
        if pattern.search(text):
            return label
    return None


def split_locality(locality: str | None) -> tuple[str | None, str | None]:
    """'1050 Ixelles' → ('Ixelles', '1050'); 'Ghent (9000)' → ('Ghent', '9000'); 'Ghent' → ('Ghent', None)."""
    if not locality:
        return None, None
    text = re.sub(r"\s+", " ", locality).strip()
    match = _POSTCODE.match(text)
    if match and match.group(2):
        city = match.group(2).strip(" ,-")
        return (city.title() if city.isupper() else city), match.group(1)
    match = _POSTCODE_END.match(text)
    if match and match.group(1):
        return match.group(1).strip(" ,-"), match.group(2)
    return text, None
