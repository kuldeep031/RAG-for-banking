import pandas as pd

from src.config import EVAL_DIR, ensure_directories


STARTER_QUESTIONS = [
    {
        "question_id": "Q001",
        "question": "If a bank does not clearly assign responsibilities to its operational risk management function, what operational risk level is indicated under RBI's 2024 operational risk guidance?",
        "question_type": "operational_risk",
        "reference_answer": "RBI's 2024 guidance expects clear responsibilities for the operational risk management function. If responsibilities are not clearly assigned, the control environment is weak and the situation indicates high operational risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q002",
        "question": "If a bank cannot continue critical operations during a major disruption, what operational risk level is indicated under RBI's operational resilience guidance?",
        "question_type": "operational_resilience",
        "reference_answer": "RBI defines operational resilience as the ability to deliver critical operations through disruption. If a bank cannot continue critical operations during disruption, the situation indicates high operational risk and weak operational resilience.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q003",
        "question": "If senior management does not receive regular reporting on material operational risk exposures and losses, what operational risk level is indicated?",
        "question_type": "operational_risk",
        "reference_answer": "RBI expects regular reporting of material operational risk exposures and losses to management and the board. Failure to ensure such reporting indicates weak governance and high operational risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q004",
        "question": "If a bank does not maintain clear written credit policies defining target markets, approval authority, and portfolio management, what credit risk level is indicated under RBI credit risk guidance?",
        "question_type": "credit_risk",
        "reference_answer": "RBI's credit risk guidance requires clear written credit policies that define target markets, risk acceptance criteria, approval authority, and portfolio management. Absence of these controls indicates high credit risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q005",
        "question": "If a bank has no independent credit risk management function and loan decisions are made without effective checks and balances, what credit risk level is indicated?",
        "question_type": "credit_risk",
        "reference_answer": "RBI guidance emphasizes an independent credit risk management function and checks and balances around credit approval. If these are absent, the situation indicates high credit risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q006",
        "question": "If a bank does not monitor concentration risk by borrower, industry, or geography, what credit risk level is indicated?",
        "question_type": "credit_risk",
        "reference_answer": "RBI expects banks to monitor risk concentrations by obligor, industry, and geography as part of credit risk management. Failure to do so indicates high credit risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q007",
        "question": "If a bank lacks a Board-approved stress testing framework and does not document stress assumptions, what risk governance level is indicated?",
        "question_type": "stress_testing",
        "reference_answer": "RBI requires banks to maintain a Board-approved stress testing framework with documented assumptions, methodologies, reporting lines, and remedial actions. Absence of these controls indicates high risk governance weakness.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q008",
        "question": "If stress test results are not reviewed by senior management or reported to the Board, what risk oversight level is indicated?",
        "question_type": "stress_testing",
        "reference_answer": "RBI stress testing guidance requires senior management review and Board reporting of stress test results. Failure to do so indicates weak oversight and high risk management concern.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q009",
        "question": "If credit models are deployed without proper governance, validation, and oversight, what model risk level is indicated under RBI's draft model-risk principles?",
        "question_type": "model_risk",
        "reference_answer": "RBI's draft principles for management of model risks in credit emphasize governance, oversight, model development discipline, and model validation. Deployment without these safeguards indicates high model risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q010",
        "question": "If a private sector bank has weak board oversight and poor governance over risk-taking, what governance-related risk level is indicated?",
        "question_type": "governance",
        "reference_answer": "RBI's governance framework for private sector banks expects sound board oversight, diversified ownership, and responsible governance over risk-taking. Weak board oversight indicates high governance-related risk.",
        "expected_risk_label": "High",
    },
    {
        "question_id": "Q011",
        "question": "What is the purpose of RBI's Guidance Note on Operational Risk Management and Operational Resilience?",
        "question_type": "operational_resilience",
        "reference_answer": "The purpose of RBI's Guidance Note is to improve the effectiveness of operational risk management and enhance operational resilience so that regulated entities can withstand, adapt to, recover from, and learn from disruptions while continuing critical operations.",
        "expected_risk_label": "Unknown",
    },
    {
        "question_id": "Q012",
        "question": "What are the key building blocks of an effective credit risk management framework according to RBI guidance?",
        "question_type": "credit_risk",
        "reference_answer": "RBI's credit risk guidance describes the key building blocks as strategy and policy, organisation, and operations or systems, supported by risk appetite, approval discipline, monitoring, MIS, and independent credit risk management.",
        "expected_risk_label": "Unknown",
    },
]


# Gold relevance must be rebuilt after extraction, chunking, and index regeneration
# because the new India-specific PDF corpus creates new chunk identifiers.
STARTER_GOLD_RELEVANCE: list[dict[str, str]] = []


UI_DEMO_QUESTIONS = [
    {
        "demo_id": "D001",
        "category": "operational_risk",
        "question": "What does RBI expect from a bank's operational risk management function?",
    },
    {
        "demo_id": "D002",
        "category": "operational_resilience",
        "question": "What does RBI mean by operational resilience for banks?",
    },
    {
        "demo_id": "D003",
        "category": "operational_resilience",
        "question": "Why is continuing critical operations during disruption important under RBI guidance?",
    },
    {
        "demo_id": "D004",
        "category": "credit_risk",
        "question": "What should a bank's credit risk policy include according to RBI?",
    },
    {
        "demo_id": "D005",
        "category": "credit_risk",
        "question": "Why does RBI emphasize an independent credit risk management function?",
    },
    {
        "demo_id": "D006",
        "category": "stress_testing",
        "question": "What should a Board-approved stress testing framework contain according to RBI?",
    },
    {
        "demo_id": "D007",
        "category": "stress_testing",
        "question": "How does stress testing support risk management in Indian banks?",
    },
    {
        "demo_id": "D008",
        "category": "model_risk",
        "question": "Why is model validation important in RBI's draft principles for management of model risks in credit?",
    },
    {
        "demo_id": "D009",
        "category": "governance",
        "question": "What does RBI expect from the board of a private sector bank in terms of governance?",
    },
    {
        "demo_id": "D010",
        "category": "comparison",
        "question": "Compare how RBI guidance addresses credit risk, operational risk, and model risk.",
    },
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
    pd.DataFrame(UI_DEMO_QUESTIONS).to_csv(EVAL_DIR / "ui_demo_questions.csv", index=False)


if __name__ == "__main__":
    create_starter_testset_files()
    print("Created India-specific starter evaluation dataset and UI demo questions.")
