from typing import List


class StepInput:
    """An input for a step"""


class StepOutput:
    """An output for a step"""


class Step:
    """A step represents a single step/stage of a pipeline.

    Classes which extend this class are eligible for use in pipelines.
    """

    def execute(self, inputs: List[StepInput]) -> List[StepOutput]:
        ...


class Pipeline:
    """A pipeline is a collection of steps or stages.
    Each steps has an input and produces the input used by the next stage as an input.
    """

    def __init__(self):
        self.steps = []

    def add_step(self, step: Step):
        self.steps.append(step)

    def execute(self):
        return [step.execute() for step in self.steps]
