from abc import ABC, abstractmethod
from model_manager import ModelManager
import logging

class BaseMethod(ABC):
    def __init__(self, args):
        self.args = args
        logging.info(f"model_name: {args.model}, model_id: {args.model_id}")
        self.manager = ModelManager(args, model_name=args.model, model_id=args.model_id)

    @abstractmethod
    def run(self, question):
        """Abstract method to be implemented by subclasses."""
        pass