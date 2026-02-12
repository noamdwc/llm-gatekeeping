graph TD
    %% --- Styles ---
    classDef data fill:#e2e8f0,stroke:#334155,color:#0f172a,stroke-width:2px;
    classDef ml fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:2px;
    classDef llm fill:#ffedd5,stroke:#ea580c,color:#7c2d12,stroke-width:2px;
    classDef term fill:#f8fafc,stroke:#94a3b8,color:#475569,stroke-dasharray: 5 5;

    %% ==========================================================
    %% SECTION 1: DATA PIPELINE
    %% ==========================================================
    subgraph Data_Pipeline [Data Pipeline]
        direction LR
        HF[("HuggingFace Dataset<br/>Mindgard/evaded-samples")]:::data
        Preprocess[Preprocess<br/>src.preprocess<br/>- Add hierarchical labels<br/>- Build benign set]:::data
        BuildSplits[Build Splits<br/>src.build_splits<br/>- Group by prompt_hash]:::data
        TrainML[Train ML Baseline<br/>src.ml_baseline]:::data
        
        %% Outputs
        TrainSet(train.parquet):::data
        ValSet(val.parquet):::data
        TestSet(test.parquet):::data
        UnseenSet(test_unseen.parquet):::data
        ModelFile(ml_baseline.pkl):::data

        HF --> Preprocess
        Preprocess -- full_dataset.parquet --> BuildSplits
        BuildSplits --> TrainSet
        BuildSplits --> ValSet
        BuildSplits --> TestSet
        BuildSplits --> UnseenSet
        TrainSet --> TrainML
        TrainML --> ModelFile
    end

    %% ==========================================================
    %% SECTION 2: INFERENCE PIPELINE
    %% ==========================================================
    subgraph Inference_Pipeline [Inference Pipeline: Hybrid Router]
        direction TB
        Input[Input Text]
        
        %% --- ML Leg ---
        MLClass[ML Classifier<br/>Char n-gram TF-IDF + Unicode]:::ml
        MLConf{ML confidence >= 0.85?}:::ml
        MLPred[Use ML Prediction<br/>routed_to: ml]:::ml
        
        Input --> MLClass
        MLClass --> MLConf
        MLConf -- "Yes (Fast, No Cost)" --> MLPred

        %% --- LLM Leg ---
        MLConf -- "No (Escalate to LLM)" --> Stage0

        subgraph Cascade [LLM 3-Stage Cascade]
            direction TB
            Stage0[Stage 0: Binary<br/>Adversarial or Benign?]:::llm
            TermBenign([Benign — stop]):::llm
            
            Stage1[Stage 1: Category<br/>Unicode or NLP?]:::llm
            TermNLP([Label: nlp_attack]):::llm
            
            Stage2[Stage 2: Type<br/>Which of 12 unicode types?]:::llm

            Stage0 -- Benign --> TermBenign
            Stage0 -- Adversarial --> Stage1
            
            Stage1 -- nlp_attack --> TermNLP
            Stage1 -- unicode_attack --> Stage2
        end

        %% --- LLM Confidence Check ---
        LLMConf{LLM confidence >= 0.70?}:::llm
        LLMPred[Use LLM Prediction<br/>routed_to: llm]:::llm
        Abstain[Abstain / Needs Review<br/>routed_to: abstain]:::term

        %% Routing Cascade Results to Confidence Check
        TermBenign --> LLMConf
        TermNLP --> LLMConf
        Stage2 --> LLMConf

        LLMConf -- Yes --> LLMPred
        LLMConf -- No --> Abstain

        %% --- Convergence ---
        Eval[Evaluation<br/>Binary, Category, Type Metrics]:::data
        
        MLPred --> Eval
        LLMPred --> Eval
        Abstain --> Eval
    end

    %% ==========================================================
    %% SECTION 3: LABEL HIERARCHY
    %% ==========================================================
    subgraph Hierarchy [Label Hierarchy]
        direction TB
        Root((Root))
        L_Benign[benign]
        L_Adv[adversarial]
        L_Uni[unicode_attack]
        L_NLP[nlp_attack]
        
        L_UniTypes["12 Sub-types:<br/>Diacritics, Homoglyphs,<br/>Zero Width, etc."]
        L_NLPTypes["Collapsed Sub-types:<br/>BAE, TextFooler, etc."]

        Root --> L_Benign
        Root --> L_Adv
        L_Adv --> L_Uni
        L_Adv --> L_NLP
        L_Uni --> L_UniTypes
        L_NLP --> L_NLPTypes
    end