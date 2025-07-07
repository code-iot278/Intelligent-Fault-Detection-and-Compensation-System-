import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from deap import base, creator, tools, algorithms
import random
from datetime import datetime

# === Define the Fuzzy Logic Controller (FLC) ===
voltage_sag = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'voltage_sag')
harmonic_distortion = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'harmonics')
control_action = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'control_action')

voltage_sag.automf(3)
harmonic_distortion.automf(3)
control_action['low'] = fuzz.trimf(control_action.universe, [0, 0, 0.5])
control_action['medium'] = fuzz.trimf(control_action.universe, [0.25, 0.5, 0.75])
control_action['high'] = fuzz.trimf(control_action.universe, [0.5, 1, 1])

rule1 = ctrl.Rule(voltage_sag['poor'] | harmonic_distortion['poor'], control_action['high'])
rule2 = ctrl.Rule(voltage_sag['average'] | harmonic_distortion['average'], control_action['medium'])
rule3 = ctrl.Rule(voltage_sag['good'] & harmonic_distortion['good'], control_action['low'])

flc_control = ctrl.ControlSystem([rule1, rule2, rule3])
flc_simulator = ctrl.ControlSystemSimulation(flc_control)

# === MPC ===
def mpc_controller(current_state, reference, horizon=5):
    A, B = 1, 0.1
    best_u, min_cost = 0, float('inf')
    for u in np.linspace(0, 1, 11):
        cost, x = 0, current_state
        for _ in range(horizon):
            x = A * x + B * u
            cost += (x - reference) ** 2
        if cost < min_cost:
            min_cost, best_u = cost, u
    return best_u

# === Genetic Algorithm ===
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: random.uniform(0, 1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_individual(ind):
    opt_sag, opt_harm, opt_ref = ind
    flc_simulator.input['voltage_sag'] = opt_sag
    flc_simulator.input['harmonics'] = opt_harm
    flc_simulator.compute()
    fuzzy_out = flc_simulator.output['control_action']
    mpc_out = mpc_controller(opt_sag, opt_ref)
    return abs(fuzzy_out - mpc_out),

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.4)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

# === Load Input CSV ===
csv_path = ''
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

results = []

for idx, row in df.iterrows():
    try:
        fuzzy_score = float(row['Fuzzy_Severity_Score'])
        init_sag, init_harm, init_ref = fuzzy_score, fuzzy_score, 1.0

        pop = toolbox.population(n=10)
        for _ in range(10):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
            fits = list(map(toolbox.evaluate, offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, k=len(pop))
        best = tools.selBest(pop, 1)[0]

        best_sag, best_harm, best_ref = best
        flc_simulator.input['voltage_sag'] = best_sag
        flc_simulator.input['harmonics'] = best_harm
        flc_simulator.compute()
        flc_out = flc_simulator.output['control_action']
        mpc_out = mpc_controller(best_sag, best_ref)
        err = abs(flc_out - mpc_out)

        results.append({
            "True_Label": row['True_Label'],
            "Predicted_Label": row['Predicted_Label'],
            "CNN_Probability": row['CNN_Probability'],
            "Fuzzy_Severity_Score": fuzzy_score,
            "Fuzzy_Severity_Label": row['Fuzzy_Severity_Label'],
            "Detected_Fault": row['Detected_Fault'],
            "Corrective_Actions": row['Corrective_Actions'],
            "Optimized_Sag_Input": round(best_sag, 4),
            "Optimized_Harmonic_Input": round(best_harm, 4),
            "Optimized_Ref": round(best_ref, 4),
            "FLC_Control_Output": round(flc_out, 4),
            "MPC_Output": round(mpc_out, 4),
            "Control_Error": round(err, 4)
        })

        print(f"✅ Row {idx+1}: FLC={flc_out:.3f}, MPC={mpc_out:.3f}, Error={err:.3f}")

    except Exception as e:
        print(f"⚠️ Row {idx+1} error: {e}")
        continue

# === Save Output CSV ===
output_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_path = f''
output_df.to_csv(output_path, index=False)
print(f"\n✅ IFPGC results saved to:\n{output_path}")