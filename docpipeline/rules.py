"""
Rule-based document classifier for the LIC proposal packet.

Taxonomy (matches the manually-curated reference folders, plus an
unidentified fallback for pages that match nothing / new document types):

    Proposal_form        - FORM NO.300 "Proposal for Insurance on Own Life"
    Proposal_enclosures  - addenda / agent report / Section 45 / declarations / NEFT
    Proposal_review_slip - dot-matrix "Proposal Review Slip (Form No:3104/OIC)"
    Medical_report       - "Medical Examiner's Report (LIC 03-001)" + clinical Q&A
    KYC_documents        - identity proof: Aadhaar / PAN / Voter / Passport / DL,
                           AND bank passbook / account-opening / statement pages
                           (banking documents are filed as KYC proof, so they
                           share this class rather than having their own)
    LIC_slip             - the small landscape LIC policy/receipt slip
    unidentified         - nothing matched

Documents are multi-page and only page 1 carries the identifying header, so
each class also has page-level content signals. OCR runs words together, so
every phrase is matched against BOTH the spaced text and a compact
(space/punct-stripped) form. Classification is by precedence: the most
document-specific signals win first.
"""
import re

CATEGORIES = [
    "Proposal_form", "Proposal_enclosures", "Proposal_review_slip",
    "Medical_report", "KYC_documents", "LIC_slip", "unidentified",
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
        # Identity proof: standalone ID CARDS (Aadhaar / PAN / Voter / Passport
        # / DL) AND bank passbook/account/statement pages. Banking documents are
        # collected as KYC proof for the policy, so they are filed together
        # here instead of in a separate Bank class.
        if self._bank(low, c, raw):
            return True
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

    def _lic_slip_text(self, low, c):
        """Dot-matrix policy fields unique to the LIC policy/receipt slip."""
        return self._has(low, c, "roc.no", "roc no", "next due", "bak_int",
                         "bak int", "d.o.m", "d.i.p", "prop.no", "prop no",
                         "mode:m", "mode :m", "t&t", "prem:", "prem :", "s=a")

    def _bank(self, low, c, raw):
        # Banking documents are filed as KYC proof (see _kyc), so this is a
        # helper for _kyc rather than a category of its own.
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
    def classify(self, text, is_wide=False):
        """Category for one page.

        is_wide: the page is a WIDE landscape strip (h/w < ~0.45, i.e. over
        2.2:1). Only the LIC policy slip has that shape. Measured after
        enhancement has cropped to content, where the separation is clean and
        wide: the slips land at h/w 0.25-0.37, while every OTHER landscape page
        -- ID cards and bank sheets, which crop down to a mildly landscape box
        -- lands at 0.59-0.95. A plain "wider than tall" test would wrongly
        claim about 45 ID-card pages per batch, so it is the ratio, not mere
        landscape-ness, that identifies the slip.

        Shape has to carry this class because these slips are routinely fed
        UPSIDE-DOWN, and enhance.py deliberately refuses low-confidence
        180-degree flips, so the slip arrives mirrored and OCR returns nothing
        usable -- exactly why it used to land in 'unidentified'.

        A wide page is still offered to the text rules first, so a wide page
        that genuinely reads as something else goes to its real class; the
        shape test only catches the slip that cannot be read.
        """
        raw = " ".join(text.split())
        low = raw.lower()
        c = _compact(low)

        if is_wide and self._lic_slip_text(low, c):
            return "LIC_slip", "LIC policy slip", {}

        if len(low) < 20:
            # A wide strip with no readable text is the mirrored slip.
            if is_wide:
                return "LIC_slip", "LIC policy slip (by shape)", {}
            return "unidentified", "Blank or illegible", {}

        if self._review_slip(low, c):
            return "Proposal_review_slip", "Proposal review slip", {}
        if self._medical(low, c):
            return "Medical_report", "Medical examiner report", {}
        if self._kyc(low, c, raw):
            return "KYC_documents", "KYC / ID document", {}
        # Form-300 and enclosure pages are full A4 portrait sheets, so a wide
        # strip is never one. Skipping these for wide pages matters because
        # _proposal_context is a deliberately loose catch-all ("premium",
        # "nominee", "policy no", "assured") and the slip carries exactly those
        # words -- an upright, readable slip was being filed as a proposal form.
        if not is_wide:
            if self._form_content(low, c):
                return "Proposal_form", "Proposal form", {}
            if self._enclosure(low, c):
                return "Proposal_enclosures", "Proposal enclosure", {}
            if self._proposal_context(low, c):
                return "Proposal_form", "Proposal form", {}
        # Nothing matched: a wide strip at this point is the slip read as
        # garbage (mirrored dot-matrix print), not an unknown document.
        if is_wide:
            return "LIC_slip", "LIC policy slip (by shape)", {}
        return "unidentified", "No rule matched", {}
