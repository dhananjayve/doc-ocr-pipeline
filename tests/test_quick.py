from finpipe.quick import quick_answer
from finpipe.rag import RAG, Answer

BANK = {"source": "combined.pdf", "facts": {"kind": "bank_statement", "accounts": [
    {"bank": "State Bank of India", "account_number": "00000012345678901", "pages": "1-2", "page_start": 1,
     "page_end": 2, "holder": "ASHA RAO", "transactions": 12, "closing_balance": "1,329.20",
     "total_debits": "2,65,000", "total_credits": "82,599.80",
     "largest_debits": [{"date": "03 Oct 2024", "amount": "1,00,000", "narration": "TO TRANSFER-INB Deposit"}],
     "largest_credits": [{"date": "28 Oct 2024", "amount": "82,000", "narration": "BY TRANSFER"}]},
    {"bank": "HDFC Bank", "account_number": "00099988877766", "pages": "3-7", "page_start": 3, "page_end": 7,
     "closing_balance": "19,758.02", "largest_debits": [], "largest_credits": []},
]}}
ITR = {"source": "itr1.pdf", "facts": {"kind": "itr", "returns": [
    {"form": "ITR-1", "assessment_year": "2023-24", "name": "ANITA SHARMA", "pages": "1-9", "page_start": 1,
     "page_end": 9, "amounts": {"total_income": "6,75,330", "refund": "40"}},
    {"form": "ITR-1", "assessment_year": "2024-25", "name": "ANITA SHARMA", "pages": "10-18", "page_start": 10,
     "page_end": 18, "amounts": {"total_income": "11,81,410", "refund": "3,400"}},
]}}


def test_account_listing_with_top_transactions():
    md = quick_answer("how many bank account, list all account with info and top debit and credit", [BANK])
    assert "**2 accounts**" in md
    assert "| 1 | State Bank of India | 00000012345678901 | ASHA RAO |" in md
    assert "**Largest debits**" in md and "1,00,000" in md
    assert "**Largest credits**" in md and "82,000" in md
    assert "no model involved" in md


def test_itr_year_from_question_and_filter():
    md = quick_answer("total income and refund for AY 2024-25?", [ITR])
    assert "AY 2024-25" in md and "11,81,410" in md and "6,75,330" not in md
    md = quick_answer("what was the refund", [ITR], year="2023-24")
    assert "40" in md and "3,400" not in md
    both = quick_answer("show total income for each year", [ITR])
    assert "| Total (taxable) income | 6,75,330 | 11,81,410 |" in both


def test_questions_needing_the_model_are_not_answered():
    for q in ["why did the balance drop in October?", "closing balance on 30/01/24",
              "what was the narration of the largest debit", "explain the refund",
              "summarise the lease terms"]:
        assert quick_answer(q, [BANK, ITR]) is None, q
    assert quick_answer("list the accounts", [ITR]) is None  # no bank facts in scope


class _Store:
    def __init__(self, entries):
        self.entries = entries

    def get(self, doc_id):
        return self.entries.get(doc_id)

    def all(self):
        return self.entries


def test_rag_answers_from_facts_unless_model_requested():
    store = _Store({"d1": BANK})

    class NoRetrieval:
        def retrieve(self, *a, **k):
            raise AssertionError("retrieval must not run for an instant answer")

    rag = RAG(NoRetrieval(), slm=None, facts_lookup=store.get, facts_all=store.all)
    events = list(rag.ask_stream("list all accounts with closing balance", doc_id="d1"))
    assert len(events) == 1 and isinstance(events[0], Answer) and events[0].route == "facts"
    assert "00099988877766" in events[0].text
    assert rag.quick("list all accounts", category="property") is None
    try:
        list(rag.ask_stream("list all accounts", doc_id="d1", use_model=True))
    except AssertionError as exc:
        assert "retrieval must not run" in str(exc)  # use_model skipped the instant answer
    else:
        raise AssertionError("use_model should have gone to retrieval")
