
import time as tm
import logging

logger = logging.getLogger(__name__)

import pandas as pd
from itertools import product
from causal_reasoning.utils.probabilities_helper import find_conditional_probability2, find_probability2
from causal_reasoning.interventional_do_calculus_algorithm.scalable_problem_column_gen import ScalarProblem
from causal_reasoning.utils.get_scalable_df import getScalableDataFrame
from causal_reasoning.causal_model import CausalModel
from causal_reasoning.utils.data_gen import generate_data_for_scale_case

def genGraph(N, M):
    scalable_input: str = "U1 -> X, U3 -> Y, "
    for i in range(1,N + 1):
        scalable_input += f"U1 -> A{i}, "
        if (i == 1):
            scalable_input += "X -> A1, "
        else:
            scalable_input += f"A{i-1} -> A{i}, "
    scalable_input += f"A{N} -> Y, "

    for i in range(1,M + 1):
        scalable_input += f"U2 -> B{i}, "
        scalable_input += f"X -> B{i}, "
        for j in range(1,N + 1):
            scalable_input += f"B{i} -> A{j}, "
            
    return scalable_input[:-2]


def true_value(N,M,y0,x0,df):
    prob = 0 
    for rlt in list(product([0, 1], repeat= 2)):
        term = 1
        term *= find_conditional_probability2(dataFrame=df,targetRealization={"Y":y0},conditionRealization={f"A{N}": rlt[0]})
        term *= find_conditional_probability2(dataFrame=df,targetRealization={f"A{N}": rlt[0]},conditionRealization= {"U1": rlt[1], "X": x0})
        term *= find_probability2(dataFrame= df, realizationDict={"U1":rlt[1]})
        prob += term
    return prob

def main():
    logging.basicConfig(level=logging.INFO)
    experiment_path = "./outputs/new_flags_experiment_results.csv"
    error_path = "./outputs/new_flags_error_log.txt"

    df = pd.DataFrame(columns=['N','M','GC_LOWER_BOUND', 'GC_UPPER_BOUND', 'GC_LOWER_BOUND_REQUIRED_ITERATIONS','GC_UPPER_BOUND_REQUIRED_ITERATIONS', 'GC_SECONDS_TAKEN', 'METHOD'])
    df.to_csv(experiment_path, index=False)

    N_M = [(5, 1), (3,1), (4,2)]#, #(1,1),(2,1),(1,2),(3,1),(4,1),(2,2),(1,3),(2,3),(4,2)]
    n_tests = 2
    for values in N_M:
        N, M = values
        for i in range(0, n_tests):
            logger.info(f"{i}th N:{N} M:{M}")

            experiments_df = pd.read_csv(experiment_path)
    
            new_row = {'N':N,'M':M,'GC_LOWER_BOUND':None,'GC_UPPER_BOUND':None,'GC_LOWER_BOUND_REQUIRED_ITERATIONS':None,'GC_UPPER_BOUND_REQUIRED_ITERATIONS':None, 'GC_SECONDS_TAKEN':None, 'METHOD':1}
            new_row_df = pd.DataFrame([new_row])

            interventionValue = 1; targetValue = 1
            scalable_df = None
            try:
                scalable_df = getScalableDataFrame(M=M, N=N)
            except Exception as e:
                logger.error(f"SCALABLE DF Error_N:{N}_M:{M}_: {e}")
                with open(error_path, 'a') as file:
                    file.write(f"SCALABLE DF Error for {i}th -- N:{N},M:{M}: {e}\n")
                continue
            try:
                start = tm.time()
                scalarProblem = ScalarProblem.buildScalarProblem(M=M, N=N, interventionValue=interventionValue, targetValue=targetValue, df=scalable_df, minimum = True)
                logger.info("MIN Problem Built")
                lower , lower_iterations = scalarProblem.solve(method=1)
                logger.info(f"Minimum Optimization N:{N}, M:{M}: Lower: {lower}, Iterations: {lower_iterations}")

                scalarProblem = ScalarProblem.buildScalarProblem(M=M, N=N, interventionValue=interventionValue, targetValue=targetValue, df=scalable_df, minimum = False)
                logger.info("MAX Problem Built")
                upper, upper_iterations = scalarProblem.solve(method=1)
                upper = -upper
                logger.info(f"Maximum Optimization N:{N}, M:{M}: Upper: {upper}, Iterations: {upper_iterations}")
                end = tm.time()
                total_time = end-start
                new_row_df['GC_LOWER_BOUND'] = lower
                new_row_df['GC_UPPER_BOUND'] = upper
                new_row_df['GC_LOWER_BOUND_REQUIRED_ITERATIONS'] = lower_iterations 
                new_row_df['GC_UPPER_BOUND_REQUIRED_ITERATIONS'] = upper_iterations
                new_row_df['GC_SECONDS_TAKEN'] = total_time
                logger.info("GC Ran")
            except Exception as e:
                logger.error(f"GC Error_N:{N}_M:{M}_: {e}")
                with open(error_path, 'a') as file:
                    file.write(f"GC Error for {i}th -- N:{N},M:{M}: {e}\n")   
            experiments_df = pd.concat([experiments_df, new_row_df], ignore_index=True)
            experiments_df.to_csv(experiment_path, index=False)
            logger.info(f"CSV updated")
    logger.info("Done")

if __name__=="__main__":
    main()
