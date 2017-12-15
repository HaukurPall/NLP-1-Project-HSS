def compute_parameter_count_ran(input_dimension, layers, vocab_size):
    return (input_dimension*input_dimension*4+input_dimension*2)*layers+vocab_size*input_dimension

print(compute_parameter_count_ran(input_dimension=650, layers=2, vocab_size=0))
