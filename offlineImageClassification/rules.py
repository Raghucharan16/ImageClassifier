"""
Rule-based document classifier for the LIC proposal packet.

Taxonomy (matches the manually-curated reference folders, plus Bank and an
unidentified fallback for pages that match nothing / new document types):

    Proposal_form        - FORM NO.300 "Proposal for Insurance on Own Life"
    Proposal_enclosures  - addenda / agent report / Section 45 / declarations / NEFT
    Proposal_review_slip - dot-matrix "Proposal Review Slip (Form No:3104/OIC)"
    Medical_report       - "Medical Examiner's Report (LIC 03-001)" + clinical Q&A
    KYC_documents         - Aadhaar / PAN / Voter / Passport / DL (standalone ID cards)
    Bank                  - passbook / account-opening KYC / statement / branch sheet
    unidentified          - nothing matched

Documents are multi-page and only page 1 carries the identifying header, so
each class also has page-level content signals. OCR runs words together, so
every phrase is matched against BOTH the spaced text and a compact
(space/punct-stripped) form. Classification is by precedence: the most
document-specific signals win first. Bank is checked before KYC so genuine
bank passbook/account pages route to Bank even though field labels overlap.
"""
import re

CATEGORIES = [
    "Proposal_form", "Proposal_enclosures", "Proposal_review_slip",
    "Medical_report", "KYC_documents", "Bank", "unidentified",
]

BANK_NAMES = [
    "state bank of india", "canara bank", "indian overseas bank", "indian bank",
    "bank of baroda", "punjab national bank", "union bank of india", "hdfc bank",
    "icici bank", "axis bank", "kotak", "central bank of india", "bank of india",
    "uco bank", "karur vysya", "city union bank", "federal bank",
    "tamilnad mercantile", "overseas bank",
]


def _compact(s):
    return re.sub(r"[^a-z0-9]", "", s)


class RuleBasedClassifier:
    def __init__(self):
        self.re_pan = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
        # Aadhaar 12-digit UID: printed as XXXX XXXX XXXX (may be masked, e.g. XXXX XXXX 1234)
        self.re_aadhaar = re.compile(r"\d{4}\s\d{4}\s\d{4}")

    @staticmethod
    def _has(low, compact, *phrases):
        for p in phrases:
            if p in low:
                return True
            cp = _compact(p)
            if len(cp) >= 5 and cp in compact:
                return True
        return False

    def _review_slip(self, low, c):
        return self._has(low, c,
            "proposal review slip", "review slip", "review sli", "proposal review",
            "3104/oic", "3104/01c", "3104oic", "suggestive underwriting",
            "signature of the underwriter", "decline cases file checked",
            "acceptance decision", "accept p-t", "reinsurance code",
            "decline serial", "bank dtls account", "clauses recommended")

    def _medical(self, low, c):
        if self._has(low, c, "medical examiner's report", "medical examiners report",
                     "lic 03-001", "lic03-001", "medical diary no",
                     "confidential medical report", "conducting your medical examination"):
            return True
        # clinical continuation pages (examiner questionnaire)
        return self._has(low, c,
            "breathlessness on exertion", "irregular heartbeat", "palpitations",
            "high cholesterol", "heart ailment", "history of chest pain",
            "electrocardiogram", "clinical examination")

    def _kyc(self, low, c, raw):
        # Standalone ID CARDS (Aadhaar / PAN / Voter / Passport / DL) only.
        # Bank passbook/account pages are NOT filed here -- see _bank.
        if self._has(low, c, "uidai", "help@uidai", "download date", "enrolment no",
                     "enrollment no", "vid :", "electoral photo identity",
                     "election commission of india", "identification auth",
                     "identity authority"):
            return True
        short = len(low) < 900
        if short and self._has(low, c, "income tax department", "income tax departmen",
                               "permanent account number", "unique identification authority",
                               "authority of india"):
            return True
        # Aadhaar card layout: DOB + gender (issuer line often mangled by OCR).
        if short and self._has(low, c, "/dob", "dob :", "/d0", "/d08", "/do8") \
                and self._has(low, c, "/male", "/female", " male", " female"):
            return True
        # Aadhaar back side: 12-digit UID (printed as XXXX XXXX XXXX). The back
        # page omits UIDAI text but always carries the masked UID number.
        # Only trusted on short pages so it does not fire on a form that happens
        # to list a 12-digit reference code.
        if short and self.re_aadhaar.search(raw):
            return True
        if short and self._has(low, c, "republic of india") and self._has(low, c, "passport"):
            return True
        if self.re_pan.search(raw) and len(low) < 900:   # PAN card (both sides -> longer)
            return True
        return False

    def _bank(self, low, c, raw):
        # Genuine bank documents: passbooks, account-opening/KYC pages,
        # statements, branch/ombudsman sheets. Most of these markers are
        # distinctive enough to accept unconditionally (never appear on the
        # LIC proposal form). "passbook"/"bank statement"/"customer id" are
        # the exception -- the form/enclosures list them as income/address
        # proof CHECKBOX OPTIONS ("2. Bank Statement", "Customer ID:____"),
        # so those three are only trusted on short pages (real bank pages are
        # short; form/enclosure pages listing them as options are long).
        if self._has(low, c,
                     "statement of account", "banking ombudsman", "account statement",
                     "generally used abbreviations", "s/d/w/h", "cif no", "cif :",
                     "mode of operation", "mode of op", "a/c opened on", "a/c open",
                     "minimum balance", "joint holder",
                     "name and address of branch", "kyc identifier", "nominee reg"):
            return True
        if len(low) < 1000 and self._has(low, c, "passbook", "bank statement", "customer id"):
            return True
        # Bare bank-brand-name pages (short, no other markers).
        return len(low) < 250 and self._has(low, c, *BANK_NAMES)

    def _form_content(self, low, c):
        # Pages that are reliably part of Form 300 (not enclosures).
        return self._has(low, c,
            "details of nominee and appointee", "nominee and appointee",
            "date of revival", "accepted at ordinary rate", "medical / non medical",
            "premium waiver benefit", "tax residency", "income tax assessee")

    def _enclosure(self, low, c):
        # Distinctive enclosure DOCUMENT titles (not shared with Form 300 body).
        return self._has(low, c,
            "addendum", "agent's confidential report", "agents confidential report",
            "moral hazard", "female lives", "female life", "categorization of plans",
            "suitability analysis", "form for suitability", "not filing itr",
            "who are not filing", "target annuity", "annuity per annum",
            "annuity/pension is opted", "if ulip is proposed",
            "did you discuss with the proposer",
            "status of previous policies", "previous policies and are you",
            "previous insurance details", "including from other insurers",
            "good health declaration", "allocation charges",
            "questionnaire for", "supplementary questionnaire",
            "nach mandate", "ecs mandate", "electronic clearing service")

    def _proposal_context(self, low, c):
        # Any LIC-proposal packet page (the Form-300 catch-all).
        return self._has(low, c,
            "proposal for insurance", "life to be assured", "life insurance corporation",
            "sum assured", "nominee", "family history", "section 45", "section 41",
            "the insurance act", "premium", "policy no", "proposer", "tax residency",
            "premium waiver", "date of commencement", "lic of india", "assured",
            "declaration", "nomination", "sum proposed")

    # -- main -----------------------------------------------------------
    def classify(self, text):
        raw = " ".join(text.split())
        low = raw.lower()
        c = _compact(low)

        if len(low) < 20:
            return "unidentified", "Blank or illegible", {}

        if self._review_slip(low, c):
            return "Proposal_review_slip", "Proposal review slip", {}
        if self._medical(low, c):
            return "Medical_report", "Medical examiner report", {}
        if self._bank(low, c, raw):
            return "Bank", "Bank document", {}
        if self._kyc(low, c, raw):
            return "KYC_documents", "KYC / ID document", {}
        if self._form_content(low, c):
            return "Proposal_form", "Proposal form", {}
        if self._enclosure(low, c):
            return "Proposal_enclosures", "Proposal enclosure", {}
        if self._proposal_context(low, c):
            return "Proposal_form", "Proposal form", {}
        return "unidentified", "No rule matched", {}
