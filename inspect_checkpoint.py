import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--file_name", help="The checkpoint file")
parser.add_argument("--tensor_name", help="The tensor name")
parser.add_argument("--all_tensors", help="Print all tensors",
                    action="store_true")
args = parser.parse_args()

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf

print_tensors_in_checkpoint_file(file_name=args.file_name, tensor_name=args.tensor_name, all_tensors=args.all_rensors)

ckpt_reader = tf.train.load_checkpoint(args.file_name)
value = ckpt_reader.get_tensor(args.tensor_name)
