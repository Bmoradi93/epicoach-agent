"""The weekly EpiCoach cycle as a Google ADK agent graph.

    pixi run -e research python -m garmin_lab.agents.cycle [--model gemini-3.8-flash] [--dry-run]

    WeeklyCycle = SequentialAgent([
        Hypothesizer (LLM)        residual report + registry -> <= 3 new/refined hypotheses
        ModelerCritic (LoopAgent) Modeler proposes a spec, Critic accepts/revises; <= 3 rounds
        Statistician (tool-only)  evaluates directional claims and refits structural specs -> record_evidence (gated)
        ExperimentDesigner (LLM)  reads EIG ranking + safety, chooses a feasible plan, pre-registers predictions
        Scribe (LLM)              weekly report with twin-computed numbers only
    ])

Only ``garmin_lab.agents.tools`` are exposed. The LLM cannot set beliefs,
compute evidence, or admit an infeasible plan: those are tool-side invariants.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types

from garmin_lab.agents import tools as T
from garmin_lab.ingest.client import REPO_ROOT

log = logging.getLogger("garmin_lab.agents")
load_dotenv(REPO_ROOT / ".env")
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

DEFAULT_MODEL = "gemini-3.8-flash"
GEN = types.GenerateContentConfig(temperature=0.0)


def exit_loop(tool_context: ToolContext) -> dict:
    """Critic calls this when the current model spec is accepted (or nothing remains to review)."""
    tool_context.actions.escalate = True
    return {"ok": True}


HYPOTHESIZER_INSTR = """You are the Hypothesizer in an epistemic-coaching system for ONE mountain biker.
You reason over structured model diagnostics and propose falsifiable, athlete-specific hypotheses.

Procedure:
1. Call athlete_snapshot, residual_report, registry_list and model_vocabulary.
2. Look for anomalies: residual patterns by covariate, segment-level residuals, autocorrelation,
   parameters whose posteriors are wide or gated (identifiability), hypotheses that are refuted.
3. Propose AT MOST 3 hypotheses. Each must be (a) specific to this athlete, (b) falsifiable with
   the available covariates and segments, (c) not already in the registry (refine instead).
   Prefer directional_claim specs on existing parameters when one exists, because they can be
   evaluated immediately; use add_covariate / add_interaction only when a genuinely new effect is
   suspected and name the covariate from the vocabulary.
4. Register each with registry_register (or registry_refine), using EXACTLY the JSON spec format shown by
   model_vocabulary (keys: base, operation, args, notes). New hypotheses get a prior in [0.3, 0.7] unless
   the literature is strong; justify it in one line. If registration fails, read the error and fix the spec
   (max 2 tries per hypothesis).
5. Finish with a short list: id, claim, mechanism (one sentence, with a physiology pointer), spec, prior.

You cannot set beliefs or compute evidence. Do not describe results; describe claims and tests."""

MODELER_INSTR = """You are the Modeler. Read the Hypothesizer's output in the conversation. For each NEW hypothesis
whose spec is NOT a directional_claim, state in <= 3 lines how the template change is identifiable
given the instruments (which segments, how many traversals, which covariate contrast). If a spec is
a directional_claim, say "evaluable now". Do not propose free-form code. If the Critic has already
accepted everything, reply "nothing to revise"."""

CRITIC_INSTR = """You are the Critic. Review the Modeler's note against these red flags:
- a new parameter not separable from an existing one given the listed covariates/instruments;
- using descent heart rate as a fitness signal;
- conditioning on information available only after the outcome;
- a duplicate of an existing registry hypothesis (call registry_list to check).
Note: model_vocabulary lists 'available_covariates' that may be ADDED; a covariate is only already in the
model if it appears under 'currently_in_model'. Do not retire an add_covariate spec for redundancy otherwise.
If a hypothesis is unacceptable, call registry_retire with the reason. When everything remaining is
acceptable (or there is nothing to review), call exit_loop. Keep the reply to <= 5 lines."""

DESIGNER_INSTR = """You are the Experiment Designer. Choose next week's training plan for the athlete.

Procedure:
1. Call athlete_snapshot and registry_list(state="") to see open hypotheses.
2. Call score_workout_menu(lambda_info=0.6) to get plans ranked by information gain vs performance.
3. Starting from the top, call safety_check(first_week_loads_json=<first_week as JSON list>, hr_target_max=165)
   until a plan is feasible. You may NOT choose an infeasible plan; if the top plans are infeasible say why
   (quote the violations) and move down the list. Deload plans are allowed when ACWR is high.
4. Call build_workout(plan_name, first_week_loads_json) for the chosen plan. Every plan contains the two protocol
   rides: a FRESH test on day 1 (after the rest day) and a LOADED test on day 5; they differ only in the training
   placed on days 2, 3, 4 and 6. The athlete fills that training with his normal rides, soccer and strength work.
5. Pre-register a prediction with ledger_predict for the instrument climb, which the athlete rides at a FIXED
   160 +- 5 bpm: target "moving_s@" + the 'instrument_climb' value from athlete_snapshot, family="normal", mu = the
   'instrument_predicted_moving_s_at_160' value from athlete_snapshot (never your own guess), sigma = its
   'instrument_predicted_sd_s', conditions = {"hr_bpm": 160, "test": "fresh"} with days_ahead = 3 and again
   conditions = {"hr_bpm": 160, "test": "loaded"} with days_ahead = 7, under the fitness hypothesis most relevant
   to this week's plan. Then pre-register the instrument DESCENT the same way: target "moving_s@" + the
   'instrument_descent' value, family="lognormal", mu = 'descent_predicted_log_mu', sigma =
   'descent_predicted_log_sd', conditions = the 'descent_conditions' value, days_ahead = 7, under the descent
   hypothesis (sleep). State in one sentence what observation would count against each hypothesis (e.g. the
   loaded test NOT being slower than the fresh one after an overload week).
6. Summarise: chosen plan, why (EIG and performance numbers from the tool, not your own), which plans were vetoed
   and by which constraint, and what the test will teach us.

Never invent numbers; every figure you quote must come from a tool result."""

SCRIBE_INSTR = """You are the Scribe. Write this week's report for the athlete in Markdown (<= 400 words):
1. What we now believe (from registry_list): each hypothesis with its belief, in plain language, one line each.
2. What the data showed this week (from residual_report): 2-4 bullets with numbers.
3. What we could NOT determine and why (identifiability), honestly.
4. Next week's plan and the experiment it contains (from the Designer's summary in the conversation).
5. The pre-registered prediction(s) and what outcome would count against which hypothesis.
Every number must come from a tool result or the conversation. Then call save_report with the markdown."""


def build_hypothesis_stage(
    model: str = DEFAULT_MODEL, preamble: str = ""
) -> list[LlmAgent | LoopAgent]:
    """Hypothesizer followed by the Modeler/Critic loop. ``preamble`` is prepended to the
    Hypothesizer instruction (the replay uses it to state the origin date and a tighter budget).
    """
    tool = FunctionTool
    hypothesizer = LlmAgent(
        name="Hypothesizer",
        model=model,
        instruction=(preamble + "\n\n" if preamble else "") + HYPOTHESIZER_INSTR,
        generate_content_config=GEN,
        tools=[
            tool(T.athlete_snapshot),
            tool(T.residual_report),
            tool(T.registry_list),
            tool(T.model_vocabulary),
            tool(T.registry_register),
            tool(T.registry_refine),
        ],
        output_key="hypothesizer_out",
    )
    modeler = LlmAgent(
        name="Modeler",
        model=model,
        instruction=MODELER_INSTR,
        generate_content_config=GEN,
        tools=[tool(T.model_vocabulary), tool(T.registry_list)],
        output_key="modeler_out",
    )
    critic = LlmAgent(
        name="Critic",
        model=model,
        instruction=CRITIC_INSTR,
        generate_content_config=GEN,
        tools=[tool(T.registry_list), tool(T.registry_retire), tool(exit_loop)],
        output_key="critic_out",
    )
    loop = LoopAgent(
        name="ModelerCritic", sub_agents=[modeler, critic], max_iterations=3
    )
    return [hypothesizer, loop]


def build_cycle(model: str = DEFAULT_MODEL) -> SequentialAgent:
    tool = FunctionTool
    hypothesizer, loop = build_hypothesis_stage(model)
    designer = LlmAgent(
        name="ExperimentDesigner",
        model=model,
        instruction=DESIGNER_INSTR,
        generate_content_config=GEN,
        tools=[
            tool(T.athlete_snapshot),
            tool(T.registry_list),
            tool(T.score_workout_menu),
            tool(T.safety_check),
            tool(T.build_workout),
            tool(T.ledger_predict),
        ],
        output_key="designer_out",
    )
    scribe = LlmAgent(
        name="Scribe",
        model=model,
        instruction=SCRIBE_INSTR,
        generate_content_config=GEN,
        tools=[tool(T.registry_list), tool(T.residual_report), tool(T.save_report)],
        output_key="scribe_out",
    )
    return SequentialAgent(
        name="WeeklyCycle", sub_agents=[hypothesizer, loop, designer, scribe]
    )


def run_statistician() -> list[dict]:
    """Tool-only step between the loop and the designer: (re)evaluate every open hypothesis.
    Directional claims use the stored fit; structural specs are compiled and refit."""
    return T.evaluate_open_hypotheses()


async def run_cycle(model: str, user_msg: str) -> dict:
    agent = build_cycle(model)
    runner = InMemoryRunner(agent=agent, app_name="epicoach")
    session = await runner.session_service.create_session(
        app_name="epicoach", user_id="athlete"
    )
    content = types.Content(role="user", parts=[types.Part(text=user_msg)])
    transcript = []
    async for ev in runner.run_async(
        user_id="athlete", session_id=session.id, new_message=content
    ):
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if p.text:
                    transcript.append({"agent": ev.author, "text": p.text})
                    print(f"\n[{ev.author}]\n{p.text}")
                elif p.function_call:
                    transcript.append(
                        {
                            "agent": ev.author,
                            "call": p.function_call.name,
                            "args": dict(p.function_call.args or {}),
                        }
                    )
                    print(
                        f"  -> {ev.author} calls {p.function_call.name}({json.dumps(dict(p.function_call.args or {}), default=str)[:160]})"
                    )
        if ev.author == "Critic" and ev.actions and ev.actions.escalate:
            stats = run_statistician()
            print(
                f"\n[Statistician] evaluated {len(stats)} directional claims: {json.dumps(stats, default=str)[:800]}"
            )
            transcript.append({"agent": "Statistician", "evidence": stats})
    s = await runner.session_service.get_session(
        app_name="epicoach", user_id="athlete", session_id=session.id
    )
    return {"state": dict(s.state), "transcript": transcript}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--message",
        default="Run this week's cycle. The athlete is available every day next week and reports no illness.",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    res = asyncio.run(run_cycle(args.model, args.message))
    T.CYCLE_DIR.mkdir(parents=True, exist_ok=True)
    import datetime as dt

    (T.CYCLE_DIR / f"transcript_{dt.date.today().isoformat()}.json").write_text(
        json.dumps(res, default=str, indent=1)
    )
    print("\nregistry after cycle:")
    for h in T.registry_list()["hypotheses"]:
        print(f"  H{h['id']} {h['belief']:.2f} {h['state']:<11s} {h['text'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
