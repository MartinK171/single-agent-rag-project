import ast
import operator
import logging
from typing import Union, Optional

class Calculator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg
        }

    def evaluate(self, expression: str) -> Optional[float]:
        try:
            # Remove whitespace and validate
            expression = ''.join(expression.split())
            if not expression:
                return None
                
            # Parse expression into AST
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate and return result
            return self._eval_node(tree.body)
            
        except Exception as e:
            self.logger.error(f"Calculation error: {str(e)}")
            return None

    def _eval_node(self, node) -> float:
        if isinstance(node, ast.Num):
            return float(node.n)
            
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in self.allowed_operators:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
                
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            
            return self.allowed_operators[type(node.op)](left, right)
            
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_node(node.operand)
            
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")