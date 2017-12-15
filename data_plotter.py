import matplotlib.pyplot as plt

filepath = "perplexities/aws_perplexities/RAN_18_50_54_batch_64_embed_300_learn_5.txt"

def get_data(filepath):
    x = []
    y = []
    with open(filepath, "r") as f:
        # for optim_steps, perplexity in f.readlines():
        for line in f.readlines():
            optim_steps, perplexity = line.split()
            print(optim_steps, perplexity)
            x.append(float(optim_steps))
            y.append(float(perplexity))

    return x, y

x, y = get_data(filepath)
margin = 20

plt.plot(x, y)
plt.axis([0, x[-1], y[-1] - margin, y[0] + margin])
plt.show()
