"""统一推理输出的固定状态值，避免不同层混用同一状态字段。"""

FINAL_INVALID_INPUT = "invalid_input"
FINAL_STRUCTURALLY_INFEASIBLE = "structurally_infeasible"
FINAL_PREDICTED = "predicted"
FINAL_INTERNAL_ERROR = "internal_error"

EVIDENCE_NOT_EVALUATED = "not_evaluated"
STEREO_NOT_EVALUATED = "not_evaluated"
