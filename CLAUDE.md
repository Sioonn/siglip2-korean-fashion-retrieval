# Project Context

## What this project is
- A short-term capstone-style ML experiment on **text-to-image retrieval for Korean fashion products**.
- Domain: Korean e-commerce (Musinsa) catalogue, top-wear category, with formal AI-generated captions and a small set of real user-style queries that diverge significantly from the captions in tone, length, and content.
- Core hypothesis: the distribution gap between catalogue captions and real user queries is the main weakness of zero-shot retrieval, and aligning the text side to the user-query distribution is the highest-leverage intervention.
- Method family: contrastive image-text model fine-tuning with parameter-efficient adapters, trained on LLM-generated user-style queries.

## What this project is NOT
- **Not aimed at a top-tier venue.** No need for novelty, comprehensive ablations, or strict fairness against state-of-the-art baselines.
- **Not a research contribution in the strong sense.** A simple, defensible "our method beats the chosen baseline on the chosen evaluation" is the bar.
- **Not a product.** No deployment, no UX, no scaling considerations.

## Constraints
- **Hard time budget**: roughly one week, ideally less. Submission is the goal; minimizing time-to-submission dominates other considerations.
- **No external API budget.** All LLM-based steps must run locally on available GPU hardware.
- **Single-GPU regime** (A6000-class). All design choices should be comfortable within this.
- **Small dataset** (~thousands of products). Methods that require large-scale data are not options.

## Decision philosophy
When evaluating any methodological choice, the user's preferences are:
1. **Speed dominates rigor** when the two conflict, as long as the choice is not obviously absurd or methodologically broken.
2. **Simpler is better.** Drop ablations, conditions, hyperparameter sweeps, and moving parts whenever they don't directly serve the "beat baseline" goal.
3. **Defensibility floor**: avoid choices that would invite an obvious, fatal critique (e.g. evaluating on the training distribution, leaking labels). Above that floor, prefer the fast path.
4. **Novelty is not a goal.** Reusing standard recipes and known-good defaults is preferred over clever ideas.
5. **Risk reduction over upside chasing.** Choose the safe variant of any decision when the upside of the riskier variant is marginal.

## How to collaborate with the user
- The user wants concrete recommendations with brief justification, not menus of options to evaluate themselves.
- For methodological forks, recommend the simpler/faster variant unless it crosses the defensibility floor.
- Surface trade-offs in one or two lines and pick a side; do not hedge.
- Korean is the working language for discussion of the project itself.
