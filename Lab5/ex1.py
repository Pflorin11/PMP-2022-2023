from pgmpy.models import BayesianNetworkLastDecision
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

player_model = BayesianNetwork(
    [
        ("CP1", "CP2"),
        ("CP1", "P1D1"),
        ("CP1", "P1D2"),
        ("CP2", "P2D"),
        ("P1D1", "P2D"),
        ("P2D", "P1D2"),
        ("CP1", "CP1GraterThenCP2"),
        ("CP2", "CP1GraterThenCP2"),
        ("CP1GraterThenCP2", "W"),
        ("P1D1", "W"),
        ("P2D", "W"),
        ("P1D2", "W")
    ]
)

cpd_CP1 = TabularCPD(
    variable="CP1", variable_card=5, values=[[0.2], [0.2], [0.2], [0.2], [0.2]]
)

cpd_CP2 = TabularCPD(
    variable="CP2",
    variable_card=5,
    values=[
        [0, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0]
    ],
    evidence=["CP1"],
    evidence_card=[5]
)

cpd_P1D1 = TabularCPD(
    variable="P1D1",
    variable_card=2,

    values=[
        [0, 0.2, 0.5, 1, 1],
        [1, 0.8, 0.5, 0, 0]
    ],
    evidence=["CP1"],
    evidence_card=[5]
)

cpd_P2D = TabularCPD(
    variable="P2D",
    variable_card=2,
    values=[
        [0.3,0,0.25,0,0.5,0.5,1,0.8,1,1],
        [0.7,1,0.75,1,0.5,0.5,0,0.2,0,0]
    ],
    evidence=["CP2", "P1D1"],
    evidence_card=[5, 2]
)

cpd_P1D2 = TabularCPD(
    variable="P1D2",
    variable_card=2,
    values=[
        [0,0,0.25,0,0.5,0,0.75,0,1,1],
        [1,1,0.75,1,0.5,1,0.25,1,0,0]
    ],
    evidence=["CP1", "P2D"],
    evidence_card=[5, 2]
)
cpd_CP1GraterThenCP2 = TabularCPD(
    variable="CP1GraterThenCP2",
    variable_card=2,
    values=[
        [1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1],
        [0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,0]
    ],
    evidence=["CP1", "CP2"],
    evidence_card=[5, 5]
)

cpd_W = TabularCPD(
    variable="W",
    variable_card=5,
    values=[
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],
        [0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0],
        [0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0]
    ],
    evidence=["CP1GraterThenCP2", "P1D1", "P2D", "P1D2"],
    evidence_card=[2, 2, 2, 2]
)

player_model.add_cpds(cpd_CP1,cpd_CP2,cpd_P1D1,cpd_P2D,cpd_P1D2,cpd_CP1GraterThenCP2,cpd_W)
infer = VariableElimination(player_model)
print(infer.query(["P1D1"], evidence={"CP1": 1}))
print(infer.query(["P2D"], evidence={"CP2": 3, "P1D1": 0}))