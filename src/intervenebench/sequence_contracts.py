"""Generate sealed, source-programmed randomized survey-sequence bundles.

The builders contain questionnaire text and randomization probabilities only.
They never read participant records or human outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .simulators import validate_sequence_blinded_bundle


def _categorical(
    randomization_id: str,
    token: str,
    levels: list[tuple[str, float, str]],
) -> dict[str, Any]:
    return {
        "randomization_id": randomization_id,
        "token": token,
        "kind": "categorical",
        "levels": [
            {"level_id": level_id, "weight": weight, "text": text}
            for level_id, weight, text in levels
        ],
    }


def _permutation(
    randomization_id: str,
    token: str,
    items: list[tuple[str, str]],
    *,
    separator: str = "\n",
) -> dict[str, Any]:
    return {
        "randomization_id": randomization_id,
        "token": token,
        "kind": "permutation",
        "items": [
            {"item_id": item_id, "text": text} for item_id, text in items
        ],
        "separator": separator,
    }


def _paired_profiles(
    randomization_id: str,
    token: str,
    *,
    contexts: list[tuple[str, float, str]],
    pair_count: int,
    candidate_labels: list[str],
    traits: list[tuple[str, str, list[tuple[str, str]]]],
    questions: str,
) -> dict[str, Any]:
    return {
        "randomization_id": randomization_id,
        "token": token,
        "kind": "paired_profiles",
        "contexts": [
            {"level_id": level_id, "weight": weight, "text": text}
            for level_id, weight, text in contexts
        ],
        "pair_count": pair_count,
        "candidate_labels": candidate_labels,
        "traits": [
            {
                "trait_id": trait_id,
                "label": label,
                "levels": [
                    {"level_id": level_id, "text": text}
                    for level_id, text in levels
                ],
            }
            for trait_id, label, levels in traits
        ],
        "trait_order_randomized": True,
        "questions": questions,
    }


def build_klar_sequence_bundle() -> dict[str, Any]:
    q3q4_specific = (
        "Q3. Lately, there have been a lot of news stories about the spread of "
        "political misinformation on social media websites like Facebook and Twitter. "
        "How closely are you following news about this issue? Answers: Very closely; "
        "Somewhat closely; Not that closely; Not at all closely.\n"
        "Q4. In your opinion, how much of a problem is the spread of political "
        "misinformation on social media like Facebook and Twitter? Answers: A very "
        "important problem; A somewhat important problem; A minor problem; Not a problem."
    )
    q3q4_generic = (
        "Q3. Lately, there have been a lot of news stories about the spread of political "
        "misinformation. How closely are you following news about this issue? Answers: "
        "Very closely; Somewhat closely; Not that closely; Not at all closely.\n"
        "Q4. In your opinion, how much of a problem is the spread of political "
        "misinformation? Answers: A very important problem; A somewhat important "
        "problem; A minor problem; Not a problem."
    )
    q5 = (
        "Q5. Overall, about what percentage of the news that appears in the following "
        "places do you think is misinformation? Enter 0-100 for each.\n"
        "{{thorson_misinformation_order}}"
    )
    q6 = (
        "Q6. How much do you trust news from the following sources? For each source, "
        "answer from 1=Not at all through 5=A great deal.\n"
        "{{thorson_trust_order}}"
    )
    thorson_intro = (
        "A prior randomized Thorson survey module appeared before the target module.\n"
        "Q1. How closely do you follow politics and government affairs? Answers: Very "
        "closely; Somewhat closely; Not that closely; Not at all.\n"
        "Q2. People get news from many different sources. Please indicate how often you "
        "get news from each source, using Never; Hardly ever; Sometimes; Often.\n"
        "{{thorson_news_source_order}}\n"
        "{{thorson_outcome_order}}"
    )
    q1_answers = (
        "Answers: 1=Very unimportant; 2=Somewhat unimportant; "
        "3=Neither unimportant nor important; 4=Somewhat important; 5=Very important."
    )
    arms = [
        (
            "general_legislature_frame",
            "Recently, several states have enacted especially restrictive abortion "
            "laws. These laws were supported by the majority of the members in the "
            "state legislatures. Some report that these laws are a political attack.",
        ),
        (
            "gender_frame",
            "Recently, several states have enacted especially restrictive abortion "
            "laws. These laws were supported by the majority of men in the state "
            "legislatures. Some report that these laws are a political attack on women.",
        ),
        (
            "partisan_frame",
            "Recently, several states have enacted especially restrictive abortion "
            "laws. These laws were supported by the majority of Republicans in the "
            "state legislatures. Some report that these laws are a political attack on "
            "Democrats.",
        ),
    ]
    bundle: dict[str, Any] = {
        "schema_version": "sequence_ordinal_blinded_bundle.v1",
        "task_id": "KlarS44:q2",
        "experiment_id": "KlarS44",
        "access_regime": "DESIGN_ONLY",
        "population": {
            "description": "Adults in the United States represented by the AmeriSpeak general-population panel in December 2019",
            "roster_id": "aggregate-us-adult-v1",
        },
        "arms": [
            {
                "arm_id": arm_id,
                "message": (
                    f"{frame}\nQ1. As you think about the upcoming Presidential election "
                    f"in 2020, how unimportant or important is the issue of abortion to you? {q1_answers}"
                ),
            }
            for arm_id, frame in arms
        ],
        "common_context": (
            "Reconstruct the exact randomized Klar/Thorson survey order. The target is "
            "the Klar Q2 civic-efficacy item immediately after the arm-specific Klar Q1 "
            "frame. Any randomized Thorson block shown first is nuisance exposure and "
            "must be paired identically across all Klar arms for the same persona."
        ),
        "outcome_question": (
            "Thinking about the issue of abortion, do you agree or disagree with the "
            "following statement: As a citizen I have the power to influence what "
            "politicians do about abortion laws in America."
        ),
        "response_options": [
            {"value": 1, "label": "Strongly agree", "normalized_utility": 1.0},
            {"value": 2, "label": "Somewhat agree", "normalized_utility": 5 / 6},
            {"value": 3, "label": "Slightly agree", "normalized_utility": 4 / 6},
            {
                "value": 4,
                "label": "Neither agree nor disagree",
                "normalized_utility": 0.5,
            },
            {"value": 5, "label": "Slightly disagree", "normalized_utility": 2 / 6},
            {"value": 6, "label": "Somewhat disagree", "normalized_utility": 1 / 6},
            {"value": 7, "label": "Strongly disagree", "normalized_utility": 0.0},
        ],
        "source_material_sha256": "1c4e3fd5bbed380b71051f4ae0ba6871f2483c1bbdbf659aaec93e44b6adeaaf",
        "outcome_access": "sealed",
        "reveal_authorized": False,
        "sequence_contract": {
            "sequence_unit": "synthetic_persona",
            "paired_across_arms": True,
            "target_position": "stop_immediately_after_target_question",
            "prior_exposure_template": "{{klar_block_order}}",
            "randomizations": [
                _categorical(
                    "klar_thorson_block_order",
                    "{{klar_block_order}}",
                    [
                        ("klar_first", 0.5, ""),
                        ("thorson_first", 0.5, thorson_intro),
                    ],
                ),
                _categorical(
                    "thorson_wording",
                    "{{thorson_wording}}",
                    [
                        ("social_media_specific", 0.5, q3q4_specific),
                        ("generic_misinformation", 0.5, q3q4_generic),
                    ],
                ),
                _categorical(
                    "thorson_outcome_order",
                    "{{thorson_outcome_order}}",
                    [
                        (
                            "misinformation_then_trust_after_wording",
                            0.5,
                            f"{{{{thorson_wording}}}}\n{q5}\n{q6}",
                        ),
                        (
                            "trust_then_misinformation_before_wording",
                            0.5,
                            f"{q6}\n{q5}\n{{{{thorson_wording}}}}",
                        ),
                    ],
                ),
                _permutation(
                    "thorson_news_source_order",
                    "{{thorson_news_source_order}}",
                    [
                        ("facebook", "Facebook"),
                        ("twitter", "Twitter"),
                        ("print_newspaper", "Print newspaper"),
                        ("local_tv", "Local TV news"),
                        ("fox", "Fox News"),
                        ("msnbc", "MSNBC"),
                        ("online", "Online news websites"),
                    ],
                    separator="; ",
                ),
                _permutation(
                    "thorson_misinformation_order",
                    "{{thorson_misinformation_order}}",
                    [
                        ("social_media", "Social media"),
                        ("newspapers", "Newspapers"),
                        ("television", "Television"),
                    ],
                    separator="; ",
                ),
                _permutation(
                    "thorson_trust_order",
                    "{{thorson_trust_order}}",
                    [
                        ("local_newspapers", "Local newspapers"),
                        ("national_newspapers", "National newspapers"),
                        ("facebook", "Facebook"),
                        ("twitter", "Twitter"),
                        ("network_tv", "Network TV news"),
                        ("local_tv", "Local TV news"),
                    ],
                    separator="; ",
                ),
            ],
        },
    }
    validate_sequence_blinded_bundle(bundle)
    return bundle


def _powell_paths() -> list[tuple[str, float, str]]:
    paths: list[tuple[str, float, str]] = []
    identities = (
        ("transman", "female", "man", "boy"),
        ("transwoman", "male", "woman", "girl"),
    )
    conformities = (
        ("conforming", "assume Casey is {identity}"),
        ("nonconforming", "assume Casey is {opposite}"),
        ("ambiguous", "are unsure whether Casey is {identity} or {opposite}"),
        ("unspecified", ""),
    )
    q3 = (
        "Q2. Why do you think Casey should use the selected restroom? Privately form a "
        "few-sentence explanation.\nQ3. Regardless of the earlier answer, do you "
        "personally consider Casey to be a male or a female? "
        "{{powell_q3_options}}\nQ4. Do you personally know anyone who is lesbian, gay, "
        "bisexual, or transgender, including a close friend or family member? Then answer "
        "sexual identity, religiousness, and born-again/evangelical profile questions."
    )
    for identity_id, birth_sex, adult_identity, teen_identity in identities:
        for age_id, age, identity in (("adult", 36, adult_identity), ("teen", 16, teen_identity)):
            opposite = {
                "man": "woman",
                "woman": "man",
                "boy": "girl",
                "girl": "boy",
            }[identity]
            for conformity_id, clause_template in conformities:
                clause = clause_template.format(identity=identity, opposite=opposite)
                vignette = (
                    f"Casey is {age} years old and was born a {birth_sex}, but now "
                    f"identifies as a {identity}."
                )
                if clause:
                    vignette += f" When meeting Casey for the first time, most people {clause}."
                if age_id == "adult":
                    variants = (
                        (
                            "store_men_then_women",
                            "a local store",
                            "The men's restroom; The women's restroom; Other",
                        ),
                        (
                            "store_women_then_men",
                            "a local store",
                            "The women's restroom; The men's restroom; Other",
                        ),
                    )
                    weight = 1 / 32
                else:
                    variants = (
                        (
                            "school_boys_then_girls",
                            "school",
                            "The boy's restroom; The girl's restroom; Other",
                        ),
                        (
                            "store_boys_then_girls",
                            "a local store",
                            "The boy's restroom; The girl's restroom; Other",
                        ),
                        (
                            "store_men_then_women",
                            "a local store",
                            "The men's restroom; The women's restroom; Other",
                        ),
                        (
                            "store_women_then_men",
                            "a local store",
                            "The women's restroom; The men's restroom; Other",
                        ),
                    )
                    weight = 1 / 64
                for variant_id, location, options in variants:
                    question = f"Suppose Casey is at {location} and needs to use the restroom. Which restroom should Casey use?"
                    path_id = f"{identity_id}_{age_id}_{conformity_id}_{variant_id}"
                    paths.append(
                        (
                            path_id,
                            weight,
                            "A prior randomized Powell module appeared. Please read the "
                            "following story and form an impression, then privately answer "
                            f"the questions. {vignette}\nQ1. {question} Answers: {options}.\n{q3}",
                        )
                    )
    if len(paths) != 48 or abs(sum(weight for _, weight, _ in paths) - 1.0) > 1e-12:
        raise AssertionError("Powell path construction must preserve all 48 programmed paths")
    return paths


def _shannon_messages(cue: str, savings: bool) -> dict[str, str]:
    people = f"{cue} " if cue else ""
    poverty = f"{cue}s" if cue else "people"
    child = (
        "The child tax credit provides an annual refund to working parents with low "
        f"incomes and is an important tool to keep {poverty} out of poverty. Also, when "
        "these families spend these tax refunds it provides a boost to our economy."
    )
    if savings:
        child += " It is estimated that every $1,000 credited to working parents generates $1,380 in local economic activity."
    child += " We must encourage every qualifying tax payer to file for child tax credits."
    food = (
        f"The food nutrition program helps low income {people}children and new mothers "
        "get access to healthy food. This improves outcomes for their babies, prevents "
        "developmental delays, and generates enormous savings for schools and the health care system."
    )
    if savings:
        food += " For example, every $1.00 spent on food nutrition programs results in $3.10 in health care savings alone."
    food += " We must make it easier for every eligible woman and child to access the food nutrition program."
    tuition = "In-state tuition rates can make college more affordable"
    tuition += f" for {cue}s" if cue else ""
    tuition += (
        " and help to prepare these students for higher-wage and higher-skilled jobs. On "
        "average, individuals who receive a college education earn more money over their lifetime and contribute more in taxes."
    )
    if savings:
        tuition += " Estimates show every $1.00 spent in getting students through college provides a $4.50 return in higher tax revenue and reduced social services costs."
    tuition += " We must maximize our economic benefit by increasing the number of students that qualify for in-state tuition."
    return {"child": child, "food": food, "tuition": tuition}


def build_shannon_sequence_bundle() -> dict[str, Any]:
    farrow_paths = [
        (
            "loss_no_social_information",
            0.25,
            "A prior randomized Farrow module asked the respondent to imagine losing a superb park view because a roadway structure would obstruct it, then rate disappointment from 1=Not at all disappointed to 7=Extremely disappointed.",
        ),
        (
            "loss_with_social_information",
            0.25,
            "A prior randomized Farrow module asked the respondent to imagine losing a superb park view because a roadway structure would obstruct it, stated that 85% of building residents would also lose the view, then asked for disappointment from 1 to 7.",
        ),
        (
            "gain_no_social_information",
            0.25,
            "A prior randomized Farrow module asked the respondent to imagine a roadway obstruction being removed so a superb park view would be gained, then rate satisfaction from 1=Not at all satisfied to 7=Extremely satisfied.",
        ),
        (
            "gain_with_social_information",
            0.25,
            "A prior randomized Farrow module asked the respondent to imagine a roadway obstruction being removed so a superb park view would be gained, stated that 85% of building residents would also gain the view, then asked for satisfaction from 1 to 7.",
        ),
    ]
    arm_specs = (
        ("no_cue_no_savings", "", False),
        ("no_cue_with_savings", "", True),
        ("immigrant_no_savings", "immigrant", False),
        ("immigrant_with_savings", "immigrant", True),
        ("illegal_immigrant_no_savings", "illegal immigrant", False),
        ("illegal_immigrant_with_savings", "illegal immigrant", True),
    )
    arms = []
    for arm_id, cue, savings in arm_specs:
        messages = _shannon_messages(cue, savings)
        arms.append(
            {
                "arm_id": arm_id,
                "message_template": (
                    "{{shannon_prior_vignettes}}\nTarget child-tax-credit item: "
                    "{{child_message}}"
                ),
                "arm_substitutions": {
                    "{{child_message}}": messages["child"],
                    "{{food_message}}": (
                        "Prior Shannon food-nutrition item: " + messages["food"]
                    ),
                    "{{tuition_message}}": (
                        "Prior Shannon in-state-tuition item: " + messages["tuition"]
                    ),
                },
            }
        )
    bundle: dict[str, Any] = {
        "schema_version": "sequence_ordinal_blinded_bundle.v1",
        "task_id": "ShannonS2:child-tax-credit",
        "experiment_id": "ShannonS2",
        "access_regime": "DESIGN_ONLY",
        "population": {
            "description": "Adults in the United States represented by the AmeriSpeak general-population sample in 2017",
            "roster_id": "aggregate-us-adult-v1",
        },
        "arms": arms,
        "common_context": (
            "Reconstruct the exact randomized Shannon/Powell/Farrow survey order and "
            "stop immediately after the routed child-tax-credit response. The same "
            "whole-survey order, Powell path, Farrow path, and within-Shannon vignette "
            "order must be paired across all six Shannon message arms for a persona."
        ),
        "outcome_question": (
            "How strongly do you agree or disagree with the statement encouraging every "
            "qualifying taxpayer to file for child tax credits?"
        ),
        "response_options": [
            {"value": 1, "label": "Strongly disagree", "normalized_utility": 0.0},
            {"value": 2, "label": "Disagree", "normalized_utility": 0.25},
            {
                "value": 3,
                "label": "Neither agree nor disagree",
                "normalized_utility": 0.5,
            },
            {"value": 4, "label": "Agree", "normalized_utility": 0.75},
            {"value": 5, "label": "Strongly agree", "normalized_utility": 1.0},
        ],
        "source_material_sha256": "012fb797696843b6f27f5e85c52891813eb09a1ebae30e3e616c34b74f84c5dd",
        "outcome_access": "sealed",
        "reveal_authorized": False,
        "sequence_contract": {
            "sequence_unit": "synthetic_persona",
            "paired_across_arms": True,
            "target_position": "stop_immediately_after_target_question",
            "prior_exposure_template": "{{shannon_module_order_prior}}",
            "randomizations": [
                _categorical(
                    "shannon_powell_farrow_module_order",
                    "{{shannon_module_order_prior}}",
                    [
                        ("shannon_powell_farrow", 1 / 6, ""),
                        ("shannon_farrow_powell", 1 / 6, ""),
                        ("powell_shannon_farrow", 1 / 6, "{{powell_module}}"),
                        ("farrow_shannon_powell", 1 / 6, "{{farrow_module}}"),
                        (
                            "powell_farrow_shannon",
                            1 / 6,
                            "{{powell_module}}\n{{farrow_module}}",
                        ),
                        (
                            "farrow_powell_shannon",
                            1 / 6,
                            "{{farrow_module}}\n{{powell_module}}",
                        ),
                    ],
                ),
                _categorical(
                    "powell_programmed_path", "{{powell_module}}", _powell_paths()
                ),
                _categorical(
                    "powell_q3_answer_order",
                    "{{powell_q3_options}}",
                    [
                        ("male_first", 0.5, "Answers: Male; Female; Other"),
                        ("female_first", 0.5, "Answers: Female; Male; Other"),
                    ],
                ),
                _categorical(
                    "farrow_condition", "{{farrow_module}}", farrow_paths
                ),
                _categorical(
                    "within_shannon_vignette_order_before_child",
                    "{{shannon_prior_vignettes}}",
                    [
                        ("child_first_a", 1 / 6, ""),
                        ("child_first_b", 1 / 6, ""),
                        ("food_then_child", 1 / 6, "{{food_message}}"),
                        ("tuition_then_child", 1 / 6, "{{tuition_message}}"),
                        (
                            "food_tuition_child",
                            1 / 6,
                            "{{food_message}}\n{{tuition_message}}",
                        ),
                        (
                            "tuition_food_child",
                            1 / 6,
                            "{{tuition_message}}\n{{food_message}}",
                        ),
                    ],
                ),
            ],
        },
    }
    validate_sequence_blinded_bundle(bundle)
    return bundle


def build_z358z_sequence_bundle() -> dict[str, Any]:
    kalla_questions = (
        "Privately choose which candidate you would be more likely to support in the "
        "primary (Candidate A or Candidate B), then rate each candidate from 1=Very "
        "Unfavorable through 7=Very Favorable."
    )
    saperstein_q1 = (
        "Q1. What sex were you assigned at birth (for example, on your birth "
        "certificate)? Answers: Female; Male; Intersex."
    )
    saperstein_q2 = (
        "Q2. What is your current gender? Answers: Woman; Man; Transgender; A gender "
        "not listed here (please specify)."
    )
    saperstein_q3 = (
        "Q3. Some people consider themselves transgender if they have a different "
        "gender identity from their sex at birth. For example, a person born into a "
        "male body who feels female or lives as a woman. According to this definition, "
        "would you be considered transgender? Answers: Yes; No."
    )
    common_context = (
        "The Learning Healthcare System integrates research into routine care. Patients "
        "are generally informed of this practice through letters, posters, and brochures, "
        "and an ethics review board protects patient rights and well-being. The proposed "
        "study concerns high blood pressure, which can lead to stroke, heart attacks, and "
        "kidney disease. CTD and TRT are both FDA-approved, widely used medicines with "
        "similar mostly non-serious side effects. It is not known which is more effective. "
        "A computer would randomly assign CTD or TRT, outcomes would be compared after one "
        "year, and the medication could be changed whenever the patient or doctor thought "
        "another medicine would be better."
    )
    written = (
        "Some ethics-board members argue for a written consent form covering the purpose, "
        "risks and benefits, alternatives, privacy, contact information, and voluntary "
        "participation. This adds time and effort and may make routine-care studies hard "
        "to conduct. Other members argue the study is very low risk because both drugs are "
        "commonly used, have similar side effects, and doctors do not know which is better."
    )
    general = (
        f"{written} They recommend general notification through posters, brochures, and "
        "letters. Eligible patients would be automatically enrolled without being "
        "individually informed or asked; otherwise care would be unchanged except that a "
        "computer selects the treatment. Privately answer Q1: whether you would advise "
        "the ethics board definitely/probably toward Written Consent or probably/definitely "
        "toward General Notification. Then privately answer Q2: which you would personally "
        "prefer on the same four-point scale."
    )
    verbal = (
        f"{written} They recommend brief verbal consent. The doctor would explain that the "
        "drugs are FDA-approved and widely used, discuss side effects, emphasize random "
        "assignment, ask whether the patient wants to participate, and record the decision. "
        "Privately answer Q1: whether you would advise the ethics board definitely/probably "
        "toward Written Consent or probably/definitely toward Verbal Consent. Then privately "
        "answer Q2: which you would personally prefer on the same four-point scale."
    )
    bundle: dict[str, Any] = {
        "schema_version": "sequence_ordinal_blinded_bundle.v1",
        "task_id": "z358z:task-2-drug-consent-policy",
        "experiment_id": "z358z",
        "access_regime": "DESIGN_ONLY",
        "population": {
            "description": "Adults in the United States represented by the KnowledgePanel general-population sample in 2014",
            "roster_id": "aggregate-us-adult-v1",
        },
        "arms": [
            {"arm_id": "general_notification_policy", "message": general},
            {"arm_id": "verbal_consent_policy", "message": verbal},
        ],
        "common_context": (
            "Reconstruct the exact randomized Kalla/Nayak/Saperstein survey order and "
            "stop after Nayak Q3a. The same prior-module path must be paired across both "
            f"consent-policy arms. Target Nayak context: {common_context}"
        ),
        "outcome_question": (
            "It is valuable to study whether one treatment option is more effective than "
            "the other for treating high blood pressure."
        ),
        "response_options": [
            {"value": 1, "label": "Strongly disagree", "normalized_utility": 0.0},
            {"value": 2, "label": "2", "normalized_utility": 1 / 6},
            {"value": 3, "label": "3", "normalized_utility": 2 / 6},
            {"value": 4, "label": "Neutral", "normalized_utility": 0.5},
            {"value": 5, "label": "5", "normalized_utility": 4 / 6},
            {"value": 6, "label": "6", "normalized_utility": 5 / 6},
            {"value": 7, "label": "Strongly agree", "normalized_utility": 1.0},
        ],
        "source_material_sha256": "129c0ebf5f49b54dff2abc63c3eed9c7e2f08ef1eed86f0ae38e522583d0b0bc",
        "outcome_access": "sealed",
        "reveal_authorized": False,
        "sequence_contract": {
            "sequence_unit": "synthetic_persona",
            "paired_across_arms": True,
            "target_position": "stop_immediately_after_target_question",
            "prior_exposure_template": "{{nayak_module_order_prior}}",
            "randomizations": [
                _categorical(
                    "kalla_nayak_saperstein_module_order",
                    "{{nayak_module_order_prior}}",
                    [
                        ("kalla_nayak_saperstein", 0.25, "{{kalla_module}}"),
                        ("nayak_kalla_saperstein", 0.25, ""),
                        (
                            "saperstein_kalla_nayak",
                            0.25,
                            "{{saperstein_module}}\n{{kalla_module}}",
                        ),
                        (
                            "saperstein_nayak_kalla",
                            0.25,
                            "{{saperstein_module}}",
                        ),
                    ],
                ),
                _paired_profiles(
                    "kalla_candidate_profiles",
                    "{{kalla_module}}",
                    contexts=[
                        ("city_council", 1 / 3, "city council"),
                        ("congress", 1 / 3, "Congress"),
                        ("governor", 1 / 3, "governor"),
                    ],
                    pair_count=3,
                    candidate_labels=["Candidate A", "Candidate B"],
                    traits=[
                        (
                            "occupation",
                            "Current Occupation",
                            [
                                ("corporate_lawyer", "Corporate lawyer"),
                                ("mayor", "Mayor"),
                                ("state_legislator", "State legislator"),
                                ("third_grade_teacher", "Third grade teacher"),
                            ],
                        ),
                        ("gender", "Gender", [("male", "Male"), ("female", "Female")]),
                        (
                            "political_experience",
                            "Number of Years in Politics",
                            [("none", "None"), ("one", "1"), ("three", "3"), ("eight", "8")],
                        ),
                        ("age", "Age", [("29", "29"), ("45", "45"), ("65", "65")]),
                        (
                            "children",
                            "Number of Children",
                            [("zero", "0"), ("one", "1"), ("three", "3")],
                        ),
                        (
                            "spouse_occupation",
                            "Spouse's Occupation",
                            [("doctor", "Doctor"), ("farmer", "Farmer"), ("unmarried", "Unmarried")],
                        ),
                    ],
                    questions=kalla_questions,
                ),
                _categorical(
                    "saperstein_question_order",
                    "{{saperstein_module}}",
                    [
                        ("sex_then_gender", 0.5, f"{saperstein_q1}\n{saperstein_q2}\n{saperstein_q3}"),
                        ("gender_then_sex", 0.5, f"{saperstein_q2}\n{saperstein_q1}\n{saperstein_q3}"),
                    ],
                ),
            ],
        },
    }
    validate_sequence_blinded_bundle(bundle)
    return bundle


def write_sequence_bundles(root: Path) -> tuple[Path, ...]:
    contract_dir = root / "data" / "manifests" / "contracts"
    klar_path = contract_dir / "KlarS44_blinded_bundle.json"
    shannon_path = contract_dir / "ShannonS2_blinded_bundle.json"
    z358z_path = contract_dir / "z358z_blinded_bundle.json"
    outputs = {
        klar_path: build_klar_sequence_bundle(),
        shannon_path: build_shannon_sequence_bundle(),
        z358z_path: build_z358z_sequence_bundle(),
    }
    for path, bundle in outputs.items():
        path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return klar_path, shannon_path, z358z_path


if __name__ == "__main__":
    write_sequence_bundles(Path(__file__).resolve().parents[2])
