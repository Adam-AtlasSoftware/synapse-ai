# Drives synapse_engine_host over its stdin/stdout pipes and checks that the engine
# works as a separate PROCESS: it streams the telemetry contract as JSON, trains via the
# coarse-grained "train" command, and survives a bad command.
# Invoked by ctest with -DHOST_EXE=... -DWORK_DIR=... -DCOMMANDS=...

execute_process(
  COMMAND "${HOST_EXE}"
  INPUT_FILE "${COMMANDS}"
  WORKING_DIRECTORY "${WORK_DIR}"
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err
  RESULT_VARIABLE rc
)

if(NOT rc EQUAL 0)
  message(FATAL_ERROR "engine host exited with ${rc}\n${err}")
endif()

# 1) It describes itself: the Topology event the dashboard lays the graph out from.
if(NOT out MATCHES "\"ev\":\"topology\"")
  message(FATAL_ERROR "no topology event in host output:\n${out}")
endif()
if(NOT out MATCHES "\"name\":\"XOR\"")
  message(FATAL_ERROR "topology did not name the XOR model:\n${out}")
endif()

# 2) It streams StepSnapshots carrying the tensors the GUI renders.
if(NOT out MATCHES "\"ev\":\"step\"")
  message(FATAL_ERROR "no step event in host output:\n${out}")
endif()
if(NOT out MATCHES "weights\\.L0")
  message(FATAL_ERROR "step snapshot missing weight tensors:\n${out}")
endif()

# 3) It actually LEARNED XOR in-process-of-its-own: 1 XOR 0 -> ~1, 0 XOR 0 -> ~0.
if(NOT out MATCHES "\"output\":\\[0\\.9")
  message(FATAL_ERROR "1 XOR 0 did not train towards 1:\n${out}")
endif()
if(NOT out MATCHES "\"output\":\\[0\\.0")
  message(FATAL_ERROR "0 XOR 0 did not train towards 0:\n${out}")
endif()

# 4) A bad command is reported, not fatal — the engine keeps serving.
if(NOT out MATCHES "\"ev\":\"error\"")
  message(FATAL_ERROR "unknown command did not produce an error event:\n${out}")
endif()

message(STATUS "engine host IPC: topology + step events OK, XOR learned, errors handled")
