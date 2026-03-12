# NotebookLM Questions

Use these after uploading the onboarding pack and a small set of raw source files.

1. Explain this repository in simple terms for a new engineer.
2. What is the main problem this codebase is solving?
3. Is this repo closer to a research pipeline or a production service? Why?
4. What should I read first if I only have 30 minutes?
5. Trace the main DVC pipeline from raw dataset to final report.
6. Explain the label hierarchy and why it matters.
7. Why does the ML baseline exclude NLP attacks from training?
8. Trace one sample from `src/preprocess.py` through `src/build_splits.py` into research artifacts.
9. Explain the difference between `src/hybrid_router.py` and `src/research.py`.
10. What is the architectural center of gravity of this repo?
11. Which files are the riskiest to modify and why?
12. What are the most important outputs under `data/processed/`?
13. Explain how config is loaded and which settings matter most.
14. Explain how `LLM_PROVIDER`, `NVIDIA_API_KEY`, and `OPENAI_API_KEY` affect runtime behavior.
15. Walk me through the LLM classifier flow, including the judge step.
16. Explain the purpose of `.cache/llm/` and how caching could confuse debugging.
17. How does the hybrid router decide between ML, LLM, and abstain?
18. What are the main failure modes when running external dataset evaluation?
19. Which tests should I read first to understand intended behavior?
20. Quiz me on the main modules in `src/`.
21. Quiz me on the DVC stages and what each one outputs.
22. What parts of the system appear well covered by tests, and what likely still needs manual verification?
23. If I want to add a new external dataset, what files do I need to touch?
24. If I want to add a small new metric, where should I start?
25. If I want to change the hybrid routing policy, what else will I likely need to update?
26. Summarize the benign data generation and validation pipeline.
27. What seems likely to confuse a new engineer in this codebase?
28. Show me a safe first task and a risky first task in this repo.
29. Which source files should I upload alongside this onboarding pack to stay grounded?
30. What concepts am I still likely missing after reading only the summaries?
