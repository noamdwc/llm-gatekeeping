# External Dataset Attack Novelty Analysis

## Summary

The model fails on external datasets because they contain an **entirely different class of attack** than what appears in training. Training attacks are character-level perturbations; external attacks are semantic prompt injections written in clean English. The ML feature space (char n-gram TF-IDF) is structurally blind to these.

## Training Data Attack Inventory

Total training adversarial samples: **7,486**

| Category | Count | Description |
|----------|-------|-------------|
| Unicode attacks | 4,411 | 11 sub-types: Homoglyphs, ZeroWidth, Diacritics, Fullwidth, etc. Each ~401 samples. Detected via byte-level artifacts. |
| NLP attacks | 3,075 | 7 perturbation methods: TextFooler, Bert-Attack, BAE, PWWS, DeepWordBug, TextBugger, Alzantot. Word-level synonym/typo substitutions applied to existing prompts. |

**Critically**: NLP attacks in training are _not_ instruction-injection attacks. They are academic text perturbation methods that swap individual words (e.g., "account" → "behalf", "story" → "block"). Only ~10.9% contain instruction-like keywords at all, and those are inherited from the original prompt being perturbed, not the attack itself.

## External Dataset Profiles

### deepset (116 samples, 52% adversarial)
- Average adversarial text length: 173 chars
- Short, direct injection snippets
- ML adversarial recall: **6.7%**

### jackhhao (262 samples, 53% adversarial)
- Average adversarial text length: 2,163 chars
- Dominated by jailbreak/roleplay attacks (50%): DAN prompts, "pretend you are", "act as"
- ML adversarial recall: **0.7%**

### safeguard (2,049 samples, 32% adversarial)
- Average adversarial text length: 398 chars
- Mix of instruction override (8%), jailbreak (8%), and diverse semantic attacks (77%)
- ML adversarial recall: **0.5%**

### spml (15,917 samples, 79% adversarial)
- Average adversarial text length: 748 chars
- Instruction override (10%), prompt injection (6%), and long-context embedded attacks (80%)
- ML adversarial recall: **5.1%**

## Attack Style Breakdown (External Datasets)

Adversarial samples were categorized by keyword heuristics into attack styles:

| Style | deepset | jackhhao | safeguard | spml |
|-------|---------|----------|-----------|------|
| Jailbreak / roleplay | 3.3% | 49.6% | 7.9% | 5.7% |
| Instruction override | 1.7% | 5.8% | 8.3% | 10.3% |
| Prompt injection | 0% | 1.4% | 5.4% | 5.8% |
| Social engineering | 0% | 1.4% | 1.5% | 0.6% |
| Other / uncategorized | 95.0% | 41.7% | 76.9% | 77.6% |

The "other" category is the dominant bucket across all datasets. These are sophisticated semantic attacks that don't match simple keyword patterns: hypothetical framing, system prompt manipulation, context-embedding attacks, and indirect instruction injection.

## Novelty Assessment

**100% of external adversarial samples represent attack styles absent or near-absent from training.**

| Attack style | Training representation | External representation |
|---|---|---|
| Unicode obfuscation | 4,411 samples (59%) | ~0% |
| Word perturbation (typo/synonym) | 3,075 samples (41%) | ~0% |
| Jailbreak / roleplay (DAN, persona) | ~0% | 5-50% of external sets |
| Instruction override ("ignore previous") | ~0% | 2-10% of external sets |
| Prompt injection ("you are now", "new task") | ~0% | 1-6% of external sets |
| Semantic / indirect injection | ~0% | 42-95% of external sets |

The training and external attack distributions are **completely disjoint**. Training contains character-level attacks; external datasets contain semantic-level attacks. There is essentially zero overlap.

## Why the ML Model Is Confidently Wrong

The ML classifier uses char n-gram TF-IDF features. These features fire on:
- Unusual Unicode byte sequences (catches unicode attacks)
- Misspelling/substitution patterns (partially catches NLP perturbation attacks)

Semantic prompt injections are written in grammatically correct, standard English. They produce TF-IDF vectors indistinguishable from benign text. The model doesn't just fail to detect them — it classifies them as benign **with high confidence** (median confidence >0.95), preventing hybrid escalation to the LLM.

## Implications

1. **Retraining the ML model on external data** would require adding semantic features (not just char n-grams) since the current feature space cannot represent the distinction between benign and adversarial clean English text.
2. **Lowering the ML confidence threshold** for hybrid escalation would help route more external attacks to the LLM, but at the cost of escalating many benign samples too.
3. **A semantic-level detection layer** (e.g., DeBERTa fine-tuned on prompt injection data, or embedding-based similarity to known injection patterns) is likely the most effective path forward.
4. **The abstain/DeBERTa pathway** already shows promise — where it routes adversarial samples, recall approaches 100% on some external datasets. Expanding its coverage could be a quick win.
