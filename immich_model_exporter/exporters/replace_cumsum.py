import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
import os
import gc

def get_tensor_type(name, graph):
    for info in graph.value_info:
        if info.name == name:
            return info.type.tensor_type.elem_type
    for info in graph.input:
        if info.name == name:
            return info.type.tensor_type.elem_type
    for info in graph.initializer:
        if info.name == name:
            return info.data_type
    # Trace through Cast node to get output type from 'to' attribute
    for node in graph.node:
        if name in node.output and node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to':
                    return attr.i
    return None

def get_constant_value(name, graph):
    for node in graph.node:
        if name in node.output and node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    return onnx.numpy_helper.to_array(attr.t)
            # Fallback if no value attr? (Shouldn't happen for Constant)
            if len(node.attribute) > 0 and node.attribute[0].name == 'value':
                 return onnx.numpy_helper.to_array(node.attribute[0].t)
    for init in graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    return None

def create_matmul_subgraph(node, graph, model):
    """
    Replaces a CumSum node with a MatMul node.
    """
    # 1. Attributes
    axis = None
    reverse = 0
    exclusive = 0
    for attr in node.attribute:
        if attr.name == 'axis':
            axis = attr.i
        elif attr.name == 'reverse':
            reverse = attr.i
        elif attr.name == 'exclusive':
            exclusive = attr.i
            
    # If axis not in attributes, check inputs
    if axis is None:
        if len(node.input) > 1:
            axis_input_name = node.input[1]
            axis_val = get_constant_value(axis_input_name, graph)
            if axis_val is not None:
                axis = int(axis_val)
                print(f"Info: Resolved axis {axis} from input {axis_input_name}")
            else:
                print(f"Warning: Could not resolve axis from input {axis_input_name}.")
                # Debug finding the node
                found = False
                for debug_node in graph.node:
                    if debug_node.name == axis_input_name:
                        found = True
                        print(f"DEBUG: Found node {debug_node.name}, Op: {debug_node.op_type}")
                        for attr in debug_node.attribute:
                            print(f"  Attr: {attr.name}")
                        break
                if not found:
                     for init in graph.initializer:
                        if init.name == axis_input_name:
                             print(f"DEBUG: Found initializer {init.name}")
                             found = True
                             break
                if not found:
                    print(f"DEBUG: Node {axis_input_name} NOT found in graph or initializers.")

    if axis is None:
        # Flatten default - we can't handle easily yet
        print(f"Error: Flatten CumSum not supported yet for {node.name}")
        return None
    
    if reverse != 0 or exclusive != 0:
        print(f"Warning: Skipping CumSum node {node.name} (reverse/exclusive not supported)")
        return None

    # 2. Input Info
    input_name = node.input[0]
    output_name = node.output[0]
    
    # Get Shape
    def get_shape(name, visited=None):
        if visited is None:
            visited = set()
        
        # Prevent infinite recursion
        if name in visited:
            return None
        visited.add(name)
        
        # 1. Check value_info
        for info in graph.value_info:
            if info.name == name:
                shape = [d.dim_value for d in info.type.tensor_type.shape.dim]
                if all(s > 0 for s in shape):
                    return shape
        
        # 2. Check graph inputs
        for info in graph.input:
            if info.name == name:
                shape = [d.dim_value for d in info.type.tensor_type.shape.dim]
                if all(s > 0 for s in shape):
                    return shape
        
        # 3. Check Initializers (often missing from value_info if not inferred)
        for init in graph.initializer:
            if init.name == name:
                return list(init.dims)
        
        # 4. Enhanced Traceback for various operations
        producer = None
        for n in graph.node:
             if name in n.output:
                 producer = n
                 break
        
        if not producer:
            return None
        
        # Element-wise operations that preserve shape
        if producer.op_type in ["Cast", "Identity", "Not", "Abs", "Relu", "Neg", 
                                "Softmax", "Sigmoid", "Tanh", "Exp", "Log",
                                "Equal", "Less", "Greater", "And", "Or", "Xor",
                                "Floor", "Ceil", "Round", "Sqrt", "Reciprocal"]:
            if len(producer.input) > 0:
                print(f"DEBUG: Tracing shape through {producer.name} ({producer.op_type})")
                return get_shape(producer.input[0], visited)
        
        # Reshape operation - try to get target shape from second input
        if producer.op_type == "Reshape":
            if len(producer.input) >= 2:
                shape_input = producer.input[1]
                shape_value = get_constant_value(shape_input, graph)
                if shape_value is not None:
                    target_shape = shape_value.tolist() if hasattr(shape_value, 'tolist') else list(shape_value)
                    # Handle -1 and 0 in reshape
                    if -1 in target_shape or 0 in target_shape:
                        # Need source shape to resolve
                        source_shape = get_shape(producer.input[0], visited)
                        if source_shape:
                            target_shape = resolve_reshape_shape(source_shape, target_shape)
                    print(f"DEBUG: Reshape {producer.name} target shape: {target_shape}")
                    if target_shape and all(s > 0 for s in target_shape):
                        return target_shape
            # Fallback: try to get from input
            print(f"DEBUG: Reshape {producer.name} - trying input shape")
            return get_shape(producer.input[0], visited)
        
        # Binary operations - typically preserve shape (broadcasting)
        if producer.op_type in ["Add", "Sub", "Mul", "Div", "Pow", "MatMul"]:
            # Try first input
            if len(producer.input) > 0:
                print(f"DEBUG: Tracing shape through {producer.name} ({producer.op_type})")
                shape = get_shape(producer.input[0], visited)
                if shape:
                    return shape
        
        return None
    
    def resolve_reshape_shape(source_shape, target_shape):
        """Resolve -1 and 0 in reshape target shape"""
        import math
        result = list(target_shape)
        source_size = math.prod(source_shape)
        
        # Replace 0 with corresponding source dimension
        for i, dim in enumerate(result):
            if dim == 0 and i < len(source_shape):
                result[i] = source_shape[i]
        
        # Resolve -1
        if -1 in result:
            known_size = math.prod([d for d in result if d > 0])
            idx = result.index(-1)
            result[idx] = source_size // known_size if known_size > 0 else 1
        
        return result

    input_shape = get_shape(input_name)
    if not input_shape:
        # Re-find producer for debug
        producer = None
        for n in graph.node:
             if input_name in n.output:
                 producer = n
                 break
                 
        if producer:
            print(f"DEBUG Trace: Producer for {input_name} is {producer.name} (Op: {producer.op_type})")
            print(f"  Inputs: {producer.input}")
            for attr in producer.attribute:
                print(f"  Attr: {attr.name}")
        else:
            print(f"DEBUG Trace: No producer found for {name} in graph nodes.")
            
        print(f"Error: Could not determine shape for {input_name}. (Traceback failed)")
        return None

    # Get Type
    input_type = get_tensor_type(input_name, graph)
    if input_type is None:
        print(f"Warning: Could not determine type for {input_name}. Defaulting to FLOAT handling.")
        input_type = TensorProto.FLOAT # Fallback

    if axis < 0:
        axis += len(input_shape)

    dim_size = input_shape[axis]
    if dim_size <= 0:
        # Error: Dimension size for axis {axis} is dynamic or unknown ({dim_size}).
        pass 

    # Special Handling for embed_positions / embeddings where inference might be [1, N] but we want CumSum on N
    # Heuristic: If dim_size is 1 (or unknown <=0) and node name relates to embeddings, and rank is 2.
    if (dim_size <= 1) and ("sub" in node.name or "embed" in node.name) and len(input_shape) == 2:
        other_dim = input_shape[1] if axis == 0 else input_shape[0]
        if other_dim > 1:
            print(f"Heuristic: Forcing CumSum on dimension {other_dim} (Axis mismatch workaround for {node.name}).")
            dim_size = other_dim
            axis = 1 # Assume seq len is usually axis 1 in (Batch, Seq) or (1, Seq)
            
    # Check again
    if dim_size <= 0:
        print(f"Error: Dimension size for axis {axis} is dynamic or unknown ({dim_size}).")
        return None

    if dim_size == 1:
        print(f"Optimization: Dim size is 1. Replacing CumSum {node.name} with Identity.")
        identity_node = onnx.helper.make_node(
            "Identity",
            inputs=[input_name],
            outputs=[output_name],
            name=f"{node.name}_identity"
        )
        return "simple", [identity_node]

    print(f"Replacing CumSum: {node.name}, Input: {input_name}, Shape: {input_shape}, Axis: {axis}, Dim: {dim_size}, Type: {input_type}")

    # 3. Generic Permute -> MatMul -> Permute Back Strategy
    # We always permute target axis to the last dimension, perform (Input @ Matrix), then permute back.
    # Input: (..., N)
    # Matrix: (N, N)
    # Output: (..., N)
    
    # 3.1. Determine Permutation
    rank = len(input_shape)
    perm = list(range(rank))
    
    # Move target axis to the end
    actual_axis = axis
    if actual_axis < 0: actual_axis += rank
    
    perm.append(perm.pop(actual_axis))
    inv_perm = [0] * rank
    for i, p in enumerate(perm):
        inv_perm[p] = i
        
    # 3.2. Generate Matrix
    # Standard CumSum (axis=-1): [x1, x2] @ [[1, 1], [0, 1]] = [x1, x1+x2]  -> Upper Triangular
    # Reverse CumSum: [x1, x2] @ [[1, 0], [1, 1]] = [x1+x2, x2] -> Lower Triangular
    # Exclusive: subtract diagonal?
    
    # Initialize with all ones
    matrix_data = np.ones((dim_size, dim_size), dtype=np.float32)
    
    if reverse == 0:
        # Standard: Upper Triangular
        matrix_data = np.triu(matrix_data)
    else:
        # Reverse: Lower Triangular
        matrix_data = np.tril(matrix_data)
        
    if exclusive == 1:
        # Exclusive: Remove diagonal (strict triangular)
        # For Upper: strictly upper
        # For Lower: strictly lower
        np.fill_diagonal(matrix_data, 0)
        
    print(f"Info: Strategy - Axis {axis} -> Move to last. Matrix Shape {matrix_data.shape}. Reverse={reverse}, Exclusive={exclusive}")

    tensor_name = f"{node.name}_constant_matrix"
    tensor = onnx.helper.make_tensor(tensor_name, TensorProto.FLOAT, matrix_data.shape, matrix_data.flatten().tolist())
    model.graph.initializer.append(tensor)
    
    new_nodes = []
    
    # Step 1: Transpose if needed
    current_input = input_name
    if perm != list(range(rank)):
        trans_out_name = f"{input_name}_transposed"
        trans_node = onnx.helper.make_node("Transpose", inputs=[input_name], outputs=[trans_out_name], name=f"{node.name}_pre_trans", perm=perm)
        new_nodes.append(trans_node)
        current_input = trans_out_name
        
    # Step 2: Cast to Float if needed
    if input_type not in [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE]:
        cast_in_name = f"{current_input}_float"
        cast_in = onnx.helper.make_node("Cast", inputs=[current_input], outputs=[cast_in_name], to=TensorProto.FLOAT, name=f"{node.name}_cast_in")
        new_nodes.append(cast_in)
        current_input = cast_in_name
        
    # Step 3: MatMul
    # Input (..., N) @ Matrix (N, N) -> (..., N)
    matmul_out_name = f"{node.name}_matmul_out"
    matmul_node = onnx.helper.make_node("MatMul", inputs=[current_input, tensor_name], outputs=[matmul_out_name], name=f"{node.name}_matmul")
    new_nodes.append(matmul_node)
    current_input = matmul_out_name
    
    # Step 4: Cast back if needed
    if input_type not in [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE]:
        cast_out_name = f"{current_input}_casted"
        cast_out = onnx.helper.make_node("Cast", inputs=[current_input], outputs=[cast_out_name], to=input_type, name=f"{node.name}_cast_out")
        new_nodes.append(cast_out)
        current_input = cast_out_name
        
    # Step 5: Transpose back if needed
    if perm != list(range(rank)):
        trans_back_node = onnx.helper.make_node("Transpose", inputs=[current_input], outputs=[output_name], name=f"{node.name}_post_trans", perm=inv_perm)
        new_nodes.append(trans_back_node)
    else:
        # If no transpose, we need to link the last output to the node's output name
        # We can't rename the last node's output easily if it's already created, so add an Identity or just rename in the definition.
        # But here, we created nodes sequentially. The last node created is `cast_out` or `matmul_node`.
        # Simplest: Add Identity if names don't match, or Modify last node output.
        # Let's Modify last node output to be `output_name`
        last_node = new_nodes[-1]
        last_node.output[0] = output_name
        
    return "simple", new_nodes

def process_model(model_path, output_path):
    print(f"\nProcessing {model_path}...")
    
    input_dir = os.path.dirname(os.path.abspath(model_path))
    input_basename = os.path.basename(model_path)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    
    original_dir = os.getcwd()
    try:
        os.chdir(input_dir)
        model = onnx.load(input_basename)
        
        try:
            onnx.load_external_data_for_model(model, ".")
            print(f"DEBUG: Loaded external data from {input_dir}")
        except Exception as e:
            print(f"DEBUG: No external data or loading failed: {e}")
        
        print(f"DEBUG: Loaded model. Nodes: {len(model.graph.node)}")
        
        try:
            inferred_model = onnx.shape_inference.infer_shapes(model)
            if len(inferred_model.graph.node) < len(model.graph.node):
                print(f"WARNING: Shape inference dropped nodes ({len(model.graph.node)} -> {len(inferred_model.graph.node)}). Discarding inferred model.")
            else:
                model = inferred_model
                print(f"DEBUG: Shape inference successful. Nodes: {len(model.graph.node)}")
        except Exception as e:
            print(f"WARNING: Shape inference failed: {e}")
    finally:
        os.chdir(original_dir)
        
    graph = model.graph
    new_graph_nodes = []
    
    print(f"DEBUG: Total nodes in graph: {len(graph.node)}")
    
    # Pre-pass: Upgrade Cast inputs to CumSum to Float to avoid INT32 Mul issues in RKNN
    print("DEBUG: Running Pre-pass to upgrade Cast nodes for CumSum compatibility...")
    cumsum_inputs = set()
    for node in graph.node:
        if node.op_type == 'CumSum':
            for inp in node.input:
                cumsum_inputs.add(inp)

    for node in graph.node:
        if node.op_type == 'Cast' and len(node.output) > 0 and node.output[0] in cumsum_inputs:
             # Check if casting to INT32 (6) or INT64 (7)
             for attr in node.attribute:
                 if attr.name == 'to' and attr.i in [6, 7]: # TensorProto.INT32=6, INT64=7
                     print(f"Pre-pass: Upgrading Cast node {node.name} (to={attr.i}) to FLOAT for CumSum.")
                     attr.i = TensorProto.FLOAT

    replaced_count = 0
    for node in graph.node:
        if node.op_type == 'CumSum':
            print(f"DEBUG: Found CumSum node: {node.name}")
            result = create_matmul_subgraph(node, graph, model)
            if result:
                rtype, data = result
                # data is a list of new nodes
                new_graph_nodes.extend(data)
                replaced_count += 1
            else:
                new_graph_nodes.append(node)
        else:
            new_graph_nodes.append(node)

    if replaced_count == 0:
        print("No CumSum nodes found or replaced.")
        save_safe(model, output_path)
        return

    # Clear old nodes and refilling
    del graph.node[:]
    graph.node.extend(new_graph_nodes)
        
    print(f"Replaced {replaced_count} CumSum nodes.")
    save_safe(model, output_path)
    print(f"Saved to {output_path}")

    # Explicit cleanup to prevent OOM
    del model
    del graph
    if 'inferred_model' in locals():
        del inferred_model
    del new_graph_nodes
    gc.collect()


def save_safe(model, output_path):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    output_basename = os.path.basename(output_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        
        if os.path.exists(output_basename):
            os.remove(output_basename)
        data_path = output_basename + ".data"
        if os.path.exists(data_path):
            os.remove(data_path)
            
        try:
            onnx.save(model, output_basename, save_as_external_data=True, all_tensors_to_one_file=True, location=data_path, size_threshold=1024, convert_attribute=False)
        except TypeError:
            onnx.save(model, output_basename, save_as_external_data=True, all_tensors_to_one_file=True, location=data_path)
    finally:
        os.chdir(original_dir)


import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Replace CumSum with MatMul in ONNX models.")
    parser.add_argument("--input", required=True, help="Path to input ONNX model")
    parser.add_argument("--output", required=True, help="Path to output ONNX model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)
        
    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    process_model(args.input, args.output)

if __name__ == "__main__":
    main()
