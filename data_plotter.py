import matplotlib.pyplot as plt

RAN_filepath = "first_offical_run_RAN_11_31_11_batch_64_embed_300_learn_5.txt"
LSTM_filepath = "first_official_run_LSTM_12_08_43_batch_64_embed_300_learn_5.txt"

def get_data(filepath):
    x = []
    y = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            optim_steps, perplexity = line.split()
            print(optim_steps, perplexity)
            if optim_steps != "FINAL":
                x.append(float(optim_steps))
                y.append(float(perplexity))

    return x, y

RAN_x, RAN_y = get_data(RAN_filepath)
LSTM_x, LSTM_y = get_data(LSTM_filepath)
margin = 20

max_x = max(LSTM_x[-1], RAN_x[-1])
max_y = max(LSTM_y[0], RAN_y[0])
min_y = min(LSTM_y[-1], RAN_y[-1])

plt.plot(RAN_x, RAN_y)
plt.plot(LSTM_x, LSTM_y)
plt.axis([0, max_x, min_y - margin, max_y + margin])
plt.show()
