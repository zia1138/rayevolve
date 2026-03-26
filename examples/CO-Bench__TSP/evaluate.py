"""Evaluator for the CO-Bench Travelling Salesman Problem."""

import sys
import math
import uuid
import json
import logging
import importlib.util
from pathlib import Path

import ray
import typer


def load_module_from_path(file_path: str | Path, unique: bool = True):
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    module_name = path.stem
    if unique:
        module_name = f"{module_name}_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def load_data(file_path):
    """
    Load TSP instances from a CO-Bench data file.

    Each line contains one instance:
        x1 y1 x2 y2 ... xN yN output t1 t2 ... tN tN+1

    Returns:
        A list of dictionaries with keys:
            - "nodes"      : list of (x, y) coordinate tuples
            - "label_tour" : reference tour as 0-indexed list (or None)
    """
    instances = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.split(" ")
            try:
                output_idx = parts.index("output")
                num_nodes = output_idx // 2
                nodes = [(float(parts[i]), float(parts[i + 1])) for i in range(0, 2 * num_nodes, 2)]
                # Reference tour: 1-indexed in file, convert to 0-indexed, drop closing node
                tour = None
                if output_idx < len(parts) - 1:
                    tour = [int(node) - 1 for node in parts[output_idx + 1 : -1]][:-1]
                instances.append({"nodes": nodes, "label_tour": tour})
            except (ValueError, IndexError):
                continue
    return instances


def tour_cost(nodes, tour):
    """Compute Euclidean tour cost."""
    cost = 0.0
    for i in range(len(tour)):
        fx, fy = nodes[tour[i]]
        tx, ty = nodes[tour[(i + 1) % len(tour)]]
        cost += math.sqrt((tx - fx) ** 2 + (ty - fy) ** 2)
    return cost


def eval_solution(nodes, tour_indices):
    """Validate and compute tour cost. Returns (cost, error_msg)."""
    n = len(nodes)
    if len(tour_indices) != n:
        return None, f"Tour length {len(tour_indices)} != {n} nodes"
    if set(tour_indices) != set(range(n)):
        return None, "Tour is not a valid permutation"
    cost = tour_cost(nodes, tour_indices)
    return cost, None


OPTIMAL_COSTS = {
    "tsp500_test_concorde.txt": [
        16.43849479258626, 16.30760609977988, 16.55368794754589, 17.0916769200107,
        16.358815620695264, 16.355575136034258, 16.468449176999673, 16.547487678806803,
        16.624118787814286, 16.875851583784797, 16.584382768436186, 16.775629024699168,
        16.625112093123217, 16.537041048883633, 16.211908886171635, 16.507889182815646,
        16.443711824038594, 16.772997858965947, 16.576148488026003, 16.644182889540385,
        16.83104599989968, 16.798687309323867, 16.64786310345603, 16.68678554471238,
        16.539765290816586, 16.158516162147357, 16.750957469266986, 16.454327423569975,
        16.437695592935125, 16.47266324558099, 16.5807314540603, 16.640030608011333,
        16.717644006541413, 16.538629003657803, 16.73424552661684, 16.702691981178777,
        16.4488503948912, 16.65158792760706, 16.21441667652796, 16.58894596771913,
        16.62425057027662, 16.411010231382186, 16.4198250548815, 16.880314028063836,
        16.654445215349824, 16.6703557900618, 16.811423319096434, 16.681548608331166,
        16.40538961977731, 16.375709814617032, 16.4755439381876, 16.352299703304702,
        16.358345088111275, 16.446260979610017, 16.479360821405024, 16.664705227172075,
        16.514514381377964, 16.703418138718607, 16.501081465067912, 16.758043371686597,
        16.529838521968927, 16.331302381910483, 16.769035549248624, 16.667247187672565,
        16.457565298893492, 16.649335805699657, 16.82614018506712, 16.938244810751787,
        16.7896287123959, 16.45162524049444, 16.60657770837926, 16.752028686357416,
        16.538134167181376, 16.419856051838476, 17.056640374302344, 16.763628081715684,
        16.76853264913112, 16.94949524434479, 16.57562195411809, 16.665389374714852,
        16.690740743946513, 16.405456340497622, 16.442597689610583, 16.801813848508267,
        16.670030108101063, 16.62938726279957, 16.23649751271661, 16.69571793825944,
        16.587558708667046, 16.32450912204972, 16.270614173517753, 16.75899873051874,
        16.803321805550524, 16.3602825442514, 16.58252109177151, 16.450516009703893,
        16.35900041167487, 16.637551343677693, 16.572893477964705, 16.73275661200808,
        16.541081653324518, 16.466516697851265, 17.021310751236744, 16.536183906712942,
        16.77678089186245, 16.35713000043851, 16.3183776670553, 16.68224023564231,
        16.672341313126555, 16.607714934366197, 16.634734868495503, 16.674511551735357,
        16.414641537953482, 16.849240225161548, 16.74452644717401, 16.50467692427514,
        16.93072503233582, 16.38341557967758, 16.610910144984917, 16.589115661773096,
        16.366818207481515, 16.599226446198887, 16.349609487246365, 16.38083156520364,
        16.732343248542644, 16.615639804768033, 16.603236295079725, 16.12821378820771,
    ],
    "tsp1000_test_concorde.txt": [
        23.180520881091528, 23.185595820967464, 23.015849671324247, 23.537607117355098,
        23.437452128607738, 23.31718378127829, 23.337815853824736, 22.98403971254625,
        23.056714372610298, 23.344826856094013, 23.204461510197465, 22.739131293587075,
        23.188355412394525, 22.89676721383878, 23.321213972552503, 23.288168535452023,
        23.40260594371496, 23.379338976209613, 23.373901670260118, 23.217316627245133,
        23.237964507712658, 23.468791280324233, 22.921856962988343, 23.10809259424775,
        23.370845238521724, 23.241556219224208, 23.348641855759727, 23.53455701244874,
        23.385399569524708, 23.324316152061755, 23.600128423871258, 22.97776918106818,
        23.23996887566731, 23.39944035075775, 23.21410580402093, 23.093180229981513,
        23.41235476581497, 22.907788976836535, 23.023973448563986, 23.38106742108426,
        23.015367118079723, 22.610650093362192, 23.728111421819854, 23.31046641124744,
        23.25381246570274, 22.889579599261864, 23.138723098665373, 23.228706227395723,
        23.420741250703944, 23.255723604641904, 23.63211466330456, 23.03074201227862,
        23.08458884685017, 23.241154659459145, 23.445330799785832, 23.315728497380498,
        23.262087203582375, 23.43107533587823, 23.020824065107902, 23.591574572456,
        23.01019854749962, 23.006394524552746, 23.117390281951273, 23.06132560795126,
        22.899650785646813, 23.17319516968116, 23.229133743009296, 23.187607300641957,
        22.83150095703399, 23.158901255572648, 23.298349320155108, 23.364983773246387,
        23.265256805650658, 23.73268837357109, 23.07144480109362, 23.202894990560697,
        23.34293044019312, 23.027139320724427, 23.005485112127072, 23.16783838686215,
        23.505726302417372, 23.002594549857108, 23.50388356372942, 23.147934207287026,
        23.149537479144914, 23.20934617772166, 23.591015529376406, 23.04614917635098,
        23.253196613627406, 23.608716670166032, 23.313874804840438, 23.14887954791675,
        23.261925104915175, 23.283273388936596, 22.869470302805432, 23.28919260955595,
        23.291061784892037, 23.26303190269252, 23.43192602385145, 22.992654709729297,
        23.53527899384453, 23.040088044723632, 23.165752550718327, 23.346603825959306,
        23.21040140495141, 23.346553301777227, 23.192654754892565, 23.30425312678073,
        23.03197099577737, 23.33672313379179, 23.209507048094107, 23.33316267340018,
        22.832592819311447, 23.47921422142005, 23.29841589882617, 22.79469376239716,
        23.437580101042798, 22.90129840984213, 23.377778449705787, 23.152730269355438,
        23.179248710299515, 23.150584655373375, 23.303559153530237, 23.567343754278223,
        23.14174465613352, 23.236813383632978, 23.178718844944385, 23.114735241004848,
    ],
}


def norm_score_for_instance(filename, idx, score):
    """Return the normalized score for a single instance, or None if it cannot be computed."""
    optimal_list = OPTIMAL_COSTS.get(filename)
    if optimal_list is None or not isinstance(score, (int, float)):
        return None
    if idx >= len(optimal_list):
        return None
    optimal = optimal_list[idx]
    if score == 0:
        return None
    return optimal / score


def get_dev():
    """Return dev instance indices per data file."""
    dev = {
        "tsp500_test_concorde.txt": list(range(0, 128, 8)),
        "tsp1000_test_concorde.txt": list(range(0, 128, 8)),
    }
    return dev


@ray.remote(num_cpus=0)
def _solve_and_eval(main_py_path, instance_id, instance):
    """Ray remote task: load candidate module and run solve + eval for a single instance."""
    try:
        candidate = load_module_from_path(main_py_path)
        solution = candidate.solve(
            instance_id=instance_id,
            nodes=instance["nodes"],
        )
        cost, err = eval_solution(instance["nodes"], solution["tour"])
        if err:
            return {"score": None, "error": err}
        return {"score": cost, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False, timeout: float = 60.0) -> dict:
    """Evaluate a candidate main.py against TSP instances.

    By default, evaluates on dev instances only.
    If test=True, evaluates on the non-dev (test) instances instead.

    Each instance is evaluated in parallel via Ray remote tasks.
    Instances that do not complete within *timeout* seconds are marked as timed out.
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)

    main_py_path = str(Path(main_py_path).resolve())
    dev = get_dev()
    data_files = sorted(data_dir.glob("tsp*_test_concorde.txt"))

    # Collect all tasks: map future -> (instance_id, filename, orig_idx)
    future_to_info = {}

    for data_file in data_files:
        filename = data_file.name
        # Skip files without optimal references (e.g. tsp10000)
        if filename not in OPTIMAL_COSTS:
            continue
        all_instances = load_data(str(data_file))
        dev_indices = set(dev.get(filename, []))

        if test:
            selected = [(i, inst) for i, inst in enumerate(all_instances) if i not in dev_indices]
        else:
            selected = [(i, inst) for i, inst in enumerate(all_instances) if i in dev_indices]

        for orig_idx, instance in selected:
            instance_id = f"{filename.replace('.txt', '')}_{orig_idx}"
            future = _solve_and_eval.remote(main_py_path, instance_id, instance)
            future_to_info[future] = (instance_id, filename, orig_idx)

    # Wait for results with timeout
    pending = list(future_to_info.keys())
    completed_results = {}  # future -> result dict

    ready, pending = ray.wait(pending, num_returns=len(pending), timeout=timeout)

    # Fetch completed results
    for future, result in zip(ready, ray.get(ready)):
        completed_results[future] = result

    # Report timed-out instances and cancel them
    for future in pending:
        instance_id, _, _ = future_to_info[future]
        ray.cancel(future, force=True)

    # Process results: one line per instance_id
    all_normed_scores = []

    for future, (instance_id, filename, orig_idx) in future_to_info.items():
        if future in completed_results:
            result = completed_results[future]
            if result["error"] is not None:
                all_normed_scores.append(0.0)
                print(f"[instance_id={instance_id}], error=true, normalized_score=0.0000, {result['error']}")
            else:
                normed = norm_score_for_instance(filename, orig_idx, result["score"])
                normed = normed if normed is not None else 0.0
                all_normed_scores.append(normed)
                print(f"[instance_id={instance_id}], error=false, normalized_score={normed:.4f}")
        else:
            all_normed_scores.append(0.0)
            print(f"[instance_id={instance_id}], error=true, normalized_score=0.0000, Timeout after {timeout}s")

    combined_score = sum(all_normed_scores) / len(all_normed_scores) if all_normed_scores else 0.0

    return {"correct": True, "error": "", "combined_score": combined_score}


app = typer.Typer()


@app.command()
def main(
    test: bool = typer.Option(False, "--test", help="Evaluate on test instances instead of dev instances"),
    timeout: float = typer.Option(60.0, "--timeout", help="Total timeout in seconds for all instances"),
):
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    result = evaluate_candidate(project_dir / "main.py", data_dir, test=test, timeout=timeout)

    print(f"Average normalized score: {result['combined_score']:.4f}")

    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    app()
