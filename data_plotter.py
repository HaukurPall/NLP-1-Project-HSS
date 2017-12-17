import matplotlib.pyplot as plt

RAN_filepath = "perplexities/first_offical_run_RAN_11_59_40_batch_5_embed_650_learn_5.txt"
LSTM_filepath = "perplexities/first_offical_run_LSTM_12_33_05_batch_5_embed_650_learn_5.txt"

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
max_y = 300
min_y = min(LSTM_y[-1], RAN_y[-1])

fig, ax = plt.subplots()

ax.plot(RAN_x, RAN_y, label='RAN')
ax.plot(LSTM_x, LSTM_y, label='LSTM')
ax.legend()
plt.title("650 unit embeddings")
plt.axis([0, max_x, min_y - margin, max_y + margin])
plt.xlabel("Optimization steps")
plt.ylabel("Perplexity")
plt.savefig("650")
