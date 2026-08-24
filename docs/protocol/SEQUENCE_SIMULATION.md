# Randomized survey-sequence simulation

## Purpose

Some eligible interventions were embedded in surveys where independently randomized modules or questions could appear before the target outcome. An isolated treatment prompt would change the exposure respondents received. InterveneBench therefore treats the full source-programmed path up to the target question as part of the simulator contract.

This protocol governs `sequence_ordinal_blinded_bundle.v1` tasks. It does not authorize outcome access or change the human intention-to-treat estimand.

## Required contract

A sequence bundle must declare:

- every randomized element that can alter exposure before the target response;
- source-programmed probabilities for categorical branches;
- complete item sets for randomized permutations;
- the exact chronological prior-exposure template;
- the target intervention and immediately preceding target-module content;
- a stop point immediately after the primary target question; and
- sealed outcome access with reveal authorization set to false.

Validation fails if tokens are missing, duplicated, unresolved, assigned nonpositive probability, or do not sum to one.

## Paired nuisance episodes

For synthetic persona `i`, deterministic seed `s_i` selects one nuisance episode `Z_i` from the frozen source-programmed distribution. The identical `Z_i` must be reused under every intervention arm:

```text
persona i + sequence Z_i + arm 1 -> predicted distribution
persona i + sequence Z_i + arm 2 -> predicted distribution
...
```

This pairing prevents Monte Carlo differences in preceding survey exposure from masquerading as treatment effects. Changing the episode between arms is a protocol violation.

For arm `a`, simulator utility is averaged over the fixed persona/episode roster:

```text
mu_S[a] = mean_i E[U(Y_hat_i,a) | persona_i, Z_i]
```

The adapter requires one complete prediction for every arm-by-episode pair. Partial grids, duplicated pairs, repaired categories, or arm-specific episode rosters are rejected.

## Current source contracts

### KlarS44

The adapter represents:

- equal-probability Klar-first and Thorson-first module order;
- both Thorson misinformation wordings;
- both Thorson Q5/Q6 versus Q3/Q4 orders; and
- the programmed permutations of the news-source, misinformation-source, and trust-source lists.

The target is Klar Q2 immediately after the arm-specific Klar Q1 frame. Klar shares fielding and respondents with SocSci210 `xtvu5`; they must remain in one split and count once for fielding-level uncertainty.

### ShannonS2

The adapter represents:

- all six Shannon/Powell/Farrow module orders;
- all six positions of the child-tax-credit item within the three Shannon vignettes;
- all 48 programmed Powell paths across target identity, age, conformity, restroom setting, and answer order;
- Powell's independent final male/female answer order; and
- all four Farrow scenario/social-information conditions.

The target is the routed child-tax-credit item. Food-nutrition and in-state-tuition messages preceding it use the same Shannon treatment group as the target arm.

### z358z

The task contract is frozen to the source drug-study cells only: general notification versus verbal consent, followed by source Q3a / SocSci task 2. The adapter represents all four programmed Kalla/Nayak/Saperstein orders. When Kalla precedes Nayak, it generates all three candidate pairs with randomized office context, trait order, and the six source-defined attributes for both candidates. When Saperstein precedes Nayak, it reproduces both randomized sex/gender question orders. The same nuisance episode is paired across both consent-policy arms, and the simulator stops immediately after Q3a.

## Cost tier

Sequence-faithful tasks are scientifically runnable but have materially longer prompts than simple text-only tasks. Shannon, Klar, and `z358z` remain outside the first five-task low-cost engineering pilot and form a prespecified higher-cost extension tier. Cost never justifies replacing them with focal-only prompts.

## Claim boundary

Sequence reconstruction preserves the fielded randomization distribution in the simulator input. It does not show that an LLM internally experiences survey carryover like a person, recover omitted participant-level realized order, establish current truth of historical claims, or create additional independent experiments. Those limitations must remain visible in reporting.
