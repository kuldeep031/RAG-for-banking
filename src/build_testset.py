import pandas as pd

from src.config import EVAL_DIR, ensure_directories


STARTER_QUESTIONS = [
    {
        "question_id": "Q001",
        "question": "A bank uses models without independent validation. Based on SR 11-7, what model risk level does this indicate?",
        "question_type": "model_risk",
        "reference_answer": "SR 11-7 treats lack of independent validation as a serious model risk weakness because validation should verify that models perform as expected and remain fit for purpose. The situation indicates high model risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q002",
        "question": "If senior management does not oversee model use and governance reporting, what model risk level is indicated under SR 11-7?",
        "question_type": "model_risk",
        "reference_answer": "SR 11-7 expects strong governance, board oversight, and regular reporting on significant model risk. Weak senior management oversight indicates high model risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q003",
        "question": "If a bank has weak change management and unclear three lines of defence, what operational risk level is indicated?",
        "question_type": "operational_risk",
        "reference_answer": "Basel operational risk guidance treats weak change management and poorly defined three lines of defence as significant control weaknesses. The situation indicates high operational risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q004",
        "question": "If a bank cannot maintain critical operations during disruption, what operational risk level is indicated under Basel operational resilience principles?",
        "question_type": "operational_resilience",
        "reference_answer": "The operational resilience principles focus on sustaining critical operations during disruption. Inability to maintain critical operations indicates high operational risk and weak resilience.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q005",
        "question": "If the board does not review credit risk strategy and policies, what credit risk level is indicated?",
        "question_type": "credit_risk",
        "reference_answer": "Basel credit risk principles assign the board responsibility for approving and periodically reviewing credit risk strategy and policies. Failure to do so indicates high credit risk governance weakness.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q006",
        "question": "If credit approval authority and accountability are unclear, what credit risk level is indicated?",
        "question_type": "credit_risk",
        "reference_answer": "The credit-granting process should define approval authority, accountability, and responsible decision making. Unclear authority and accountability indicate high credit risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q007",
        "question": "If loan files are outdated and credit monitoring is weak, what credit risk level is indicated?",
        "question_type": "credit_risk",
        "reference_answer": "Credit administration requires current borrower information, updated documentation, and effective monitoring. Outdated files and weak monitoring indicate medium to high credit risk; for this benchmark it should be treated as High.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q008",
        "question": "If internal controls do not ensure compliance with operational risk policies, what operational risk level is indicated?",
        "question_type": "operational_risk",
        "reference_answer": "Operational risk guidance expects internal controls to ensure compliance with policies and procedures. Failure of those controls indicates high operational risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q009",
        "question": "What does SR 11-7 say model validation should verify?",
        "question_type": "model_risk",
        "reference_answer": "SR 11-7 states that model validation should verify that models are performing as expected, consistent with their design objectives and intended business uses.",
        "expected_risk_label": "Unknown",
    },
    {
        "question_id": "Q010",
        "question": "What are the three lines of defence in operational risk management?",
        "question_type": "operational_risk",
        "reference_answer": "The three lines of defence separate responsibility among business line management as the first line, independent corporate operational risk management as the second line, and independent review such as internal audit as the third line.",
        "expected_risk_label": "Unknown",
    },
    {
        "question_id": "Q011",
        "question": "What is the purpose of operational resilience principles?",
        "question_type": "operational_resilience",
        "reference_answer": "The principles aim to strengthen a bank's ability to deliver critical operations through disruption and improve resilience to operational risk events.",
        "expected_risk_label": "Unknown",
    },
    {
        "question_id": "Q012",
        "question": "Why are bank examinations important according to FDIC guidance?",
        "question_type": "supervisory_guidance",
        "reference_answer": "FDIC examination guidance explains that examinations support supervisory oversight by identifying weaknesses, assessing risk, and prompting corrective action before problems become more serious.",
        "expected_risk_label": "Unknown",
    },
]


STARTER_GOLD_RELEVANCE = [
    {"question_id": "Q001", "relevant_chunk_id": "sr1107a1_p2_c1"},
    {"question_id": "Q001", "relevant_chunk_id": "sr1107a1_p9_c1"},
    {"question_id": "Q002", "relevant_chunk_id": "sr1107a1_p16_c2"},
    {"question_id": "Q002", "relevant_chunk_id": "sr1107a1_p17_c1"},
    {"question_id": "Q003", "relevant_chunk_id": "bcbs292_p37_c1"},
    {"question_id": "Q003", "relevant_chunk_id": "bcbs292_p8_c1"},
    {"question_id": "Q004", "relevant_chunk_id": "d516_p7_c1"},
    {"question_id": "Q004", "relevant_chunk_id": "d516_p8_c1"},
    {"question_id": "Q005", "relevant_chunk_id": "d595_p6_c1"},
    {"question_id": "Q005", "relevant_chunk_id": "d595_p7_c1"},
    {"question_id": "Q006", "relevant_chunk_id": "d595_p12_c2"},
    {"question_id": "Q006", "relevant_chunk_id": "d595_p12_c1"},
    {"question_id": "Q007", "relevant_chunk_id": "d595_p14_c2"},
    {"question_id": "Q007", "relevant_chunk_id": "d595_p13_c2"},
    {"question_id": "Q008", "relevant_chunk_id": "bcbs292_p47_c1"},
    {"question_id": "Q008", "relevant_chunk_id": "bcbs292_p57_c1"},
    {"question_id": "Q009", "relevant_chunk_id": "sr1107a1_p9_c1"},
    {"question_id": "Q009", "relevant_chunk_id": "sr1107a1_p10_c2"},
    {"question_id": "Q010", "relevant_chunk_id": "bcbs292_p56_c2"},
    {"question_id": "Q010", "relevant_chunk_id": "bcbs292_p37_c1"},
    {"question_id": "Q011", "relevant_chunk_id": "d516_p7_c1"},
    {"question_id": "Q011", "relevant_chunk_id": "d516_p7_c2"},
    {"question_id": "Q012", "relevant_chunk_id": "section1_1_p2_c1"},
    {"question_id": "Q012", "relevant_chunk_id": "section1_1_p16_c2"},
]


def create_empty_testset_files() -> None:
    ensure_directories()

    questions = pd.DataFrame(
        columns=[
            "question_id",
            "question",
            "question_type",
            "reference_answer",
            "expected_risk_label",
        ]
    )
    relevance = pd.DataFrame(columns=["question_id", "relevant_chunk_id"])

    questions.to_csv(EVAL_DIR / "questions.csv", index=False)
    relevance.to_csv(EVAL_DIR / "gold_relevance.csv", index=False)


def create_starter_testset_files() -> None:
    ensure_directories()
    pd.DataFrame(STARTER_QUESTIONS).to_csv(EVAL_DIR / "questions.csv", index=False)
    pd.DataFrame(STARTER_GOLD_RELEVANCE).to_csv(EVAL_DIR / "gold_relevance.csv", index=False)


if __name__ == "__main__":
    create_starter_testset_files()
    print("Created starter evaluation dataset.")
