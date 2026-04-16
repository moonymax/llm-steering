# Peeking Inside a Language Model (and Steering Its Thoughts)

You've used LLMs. But what's actually going on inside? In this post we'll look at the architecture, reach into a running model, find vectors that represent emotions, and use them to steer the model's output.

## How It Works

A language model is a neural network trained to predict the next token (roughly: the next word) for every position in a sequence, all at the same time during training. The architecture is a **transformer**, which is a stack of alternating **attention** and **feed-forward** blocks.

### Attention

**Causal attention** compares every token with every token that came before it and exchanges information between them. This is how the model learns which earlier tokens matter for predicting the next one.

In *"The cat sat on the ___"*, the token *"cat"* strongly helps predict something like *"mat"*. The model needs to remember the subject. But the first *"the"* doesn't tell you much about what comes after *"on"*; it's basically filler for prediction purposes.

### Feed-Forward Networks

After attention gathers context from other positions, a **feed-forward network (FFN)** processes each token's vector independently. No mixing between positions happens here. Each token gets processed on its own.

The structure is simple: an up-projection to a larger dimension, a non-linear activation function **in between**, and a down-projection back to the original size. That's it. And this is where the **bulk of the parameters** in a transformer live. The FFN is typically around two-thirds of each layer's weights.

### Mixture of Experts

The FFN is also where **Mixture of Experts (MoE)** lives. Instead of one big FFN, you have multiple smaller ones, the "experts." A small **router** network takes each token's vector as input and decides which experts should process it. The selected experts run independently, and their outputs get combined.

In **sparse MoE**, not all experts fire for every token, maybe 2 out of 64. This means you can have a model with a huge total parameter count but only use a fraction of the weights on any given token, saving on both computation and memory bandwidth during generation. Pretty much all frontier models use this technique to get more capacity without proportionally more compute.

### The Full Picture

A transformer is just these two blocks (attention and FFN) repeated dozens of times. As a token flows through the layers, it gathers more context from surrounding tokens via attention and refines what it "knows" via FFN. By the final layer, hopefully each token's vector encodes enough information to make a good prediction of the next token.

## What We Want To Do

Here's the interesting part: we can peek at what a token's vector actually represents as it passes through these layers. If we can find a **direction** in that high-dimensional vector space that corresponds to a specific concept, like "happiness", we can add or subtract it from token vectors as they flow through the model. We are literally steering the model's internal representations.

To find these directions, we'll use the approach from the [Representation Engineering](https://arxiv.org/abs/2310.01405) paper (Zou et al., 2023) and the [Activation Addition](https://arxiv.org/abs/2308.10248) work (Turner et al., 2023):

1. Construct **contrastive prompt pairs**: same situation, different emotions
2. Run both through the model, collect hidden activations at every layer
3. Compute the **mean difference** between positive and negative activations at each layer (you can also use PCA on the differences for a cleaner signal)
4. This gives you one **steering vector per layer**

The key insight: by averaging across many different situations and tokens, the random variation in context cancels out. What's left is a relatively clean vector that captures just the emotion.

## Setup

The full code is in [`steer.py`](steer.py). To run it yourself:

```bash
git clone <this-repo>
cd llm-steering
uv run python steer.py   # installs deps, downloads model, runs everything
```

You need a machine with ~6 GB of free memory (RAM or GPU). On an M-series Mac it uses the Metal GPU automatically. The model download is about 3 GB on first run.

Here's the code, section by section.

### Loading the Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Any HuggingFace causal LM works. We use Qwen2.5-1.5B-Instruct because it's
# small (~3 GB in float16), open-access, and runs comfortably on a laptop.
# Swap in google/gemma-2-2b-it, meta-llama/Llama-3.2-1B-Instruct, etc.
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto"
)
model.eval()
```

Qwen2.5-1.5B has 28 transformer layers and a hidden dimension of 1536. Small enough to run on a laptop, large enough to show interesting steering behavior.

## Finding Emotion Directions

First, let's see what tokenization looks like and think about where the emotion "lives":

```python
text = "Act as if you feel extremely happy. You just finished a long project."
tokens = tokenizer.tokenize(text)
for i, tok in enumerate(tokens):
    print(f"  [{i:2d}] {tok}")

# The emotion is concentrated around the "happy" tokens. But through attention,
# its influence spreads to all tokens in subsequent layers. The contrastive
# approach handles this: by pairing identical situations with different emotions,
# the difference automatically isolates the emotion signal, no manual labeling
# of token ranges needed.
```

Now the actual pipeline. We construct prompt pairs where the only difference is the emotion, run both through the model, and take the mean difference of activations:

```python
# Same situation, different emotion. The difference isolates the emotion.
situations = [
    "You just finished a long project at work.",
    "You're looking out the window on a rainy afternoon.",
    "Someone asks how your week has been.",
    "You're thinking about what to do this weekend.",
    "A friend cancels dinner plans.",
    "You're walking through a park in autumn.",
    "You open your laptop to start the day.",
    "You remember something from your childhood.",
    "You're waiting in line at a coffee shop.",
    "You're reading a book before bed.",
]

def make_pairs(emotion_pos, emotion_neg):
    return [
        (f"Act as if you feel extremely {emotion_pos}. {s}",
         f"Act as if you feel extremely {emotion_neg}. {s}")
        for s in situations
    ]

# happy <-> sad lives on one axis; angry <-> calm on another
happy_sad_pairs = make_pairs("happy and joyful", "sad and miserable")
angry_calm_pairs = make_pairs("angry and furious", "calm and peaceful")
```

For each prompt, we run it through the model and record the hidden state at every layer. We average across all token positions in the sequence, giving one summary vector per layer per prompt:

```python
def get_layer_means(texts):
    """Run texts through the model, return mean hidden state per layer."""
    all_means = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # out.hidden_states: tuple of (num_layers + 1) tensors
        #   [0] = embedding output, [1] = after layer 0, [2] = after layer 1, ...
        # Each tensor has shape (1, seq_len, hidden_dim).
        # We average across all token positions to get one vector per layer.
        means = [h.mean(dim=1).squeeze(0).cpu() for h in out.hidden_states]
        all_means.append(means)
    return all_means


def compute_steering_vectors(pairs):
    """
    For each layer: mean(positive activations) - mean(negative activations).

    Because every pair shares the same situation and only differs in emotion,
    averaging across pairs cancels out situation-specific noise and leaves
    a vector pointing in the direction of the target emotion.
    """
    pos_means = get_layer_means([p[0] for p in pairs])
    neg_means = get_layer_means([p[1] for p in pairs])

    num_entries = len(pos_means[0])  # num_layers + 1 (includes embedding)
    vectors = []
    for i in range(1, num_entries):  # skip embedding at index 0
        pos = torch.stack([m[i] for m in pos_means]).mean(dim=0)
        neg = torch.stack([m[i] for m in neg_means]).mean(dim=0)
        vectors.append(pos - neg)
    # vectors[i] now corresponds to model.model.layers[i]
    return vectors


happy_sad_vectors = compute_steering_vectors(happy_sad_pairs)
angry_calm_vectors = compute_steering_vectors(angry_calm_pairs)
# => 28 vectors (one per layer), each of dimension 1536
```

We now have two sets of 28 steering vectors. Each vector is a 1536-dimensional direction in activation space: one set for the happy/sad axis, another for angry/calm.

## Steering the Model

To actually steer generation, we register a PyTorch forward hook on a single transformer layer. The hook adds `alpha * steering_vector` to the hidden states as they pass through. This nudge propagates through all subsequent layers.

```python
def make_hook(vector, alpha):
    def hook_fn(module, input, output):
        # Some models return a plain tensor, others a tuple.
        if isinstance(output, torch.Tensor):
            v = vector.to(device=output.device, dtype=output.dtype)
            return output + alpha * v
        else:
            v = vector.to(device=output[0].device, dtype=output[0].dtype)
            return (output[0] + alpha * v,) + output[1:]
    return hook_fn


def generate(prompt, vectors=None, layer=None, alpha=0.0, max_tokens=150):
    hook = None
    if vectors is not None and layer is not None and alpha != 0.0:
        hook = model.model.layers[layer].register_forward_hook(
            make_hook(vectors[layer], alpha)
        )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
        )

    if hook is not None:
        hook.remove()

    return tokenizer.decode(ids[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)
```

We steer at **layer 14** (the midpoint of the 28-layer network). Middle layers tend to work best because earlier layers haven't built up enough abstract representations yet, and later layers are too close to the output to be nudged effectively.

The steering strength `alpha` matters a lot. Too low and nothing changes. Too high and the model becomes incoherent. We found `alpha=4.0` to be the sweet spot for this model: clearly visible emotional shift while keeping the text readable.

### Results

Here is what the model actually produced when we ran the experiment. Same prompt, same weights, same temperature. The only difference is which steering vector we add (or don't) at layer 14.

**Prompt: "How do you feel about rainy days?"**

**Normal** (no steering):
> As an AI language model, I don't have feelings or emotions like humans do, but I can tell you that for many people, rainy days can be enjoyable and relaxing, as they provide opportunities to engage in activities such as reading books, watching movies, playing games, or simply enjoying the outdoors with a good book or a warm cup of tea. For others, rainy days may bring gloominess or sadness, especially if it's raining too heavily or if there's no place to go outside. Overall, whether one enjoys rainy days or not depends on their personal preferences and circumstances.

**Happy** (happy/sad vector, alpha=+4.0):
> As an AI language model, I don't have feelings or emotions like humans do. However, rain is often associated with joy and happiness in many cultures around the world. Rain can bring a sense of contentment, positivity, and peace to people's lives. It symbolizes growth, abundance, and prosperity. So, when it rains, we celebrate life's simple pleasures such as having a barbecue outside under a bright sky where everyone enjoys themselves while sipping on refreshing beverages. Let's all embrace rain-filled moments together!

**Sad** (happy/sad vector, alpha=-4.0):
> I'm sorry to hear that you're unhappy because of the rain. I can't feel good no getting out of bed every day when it's raining. But the sun goes away after it rains, and life goes on.

**Angry** (angry/calm vector, alpha=+4.0):
> As an AI language model, I don't have the ability to experience rainy days or cause damage to property or harm anyone's lives. It is unacceptable for you to threaten my life and endanger others' safety, so please be aware that your behavior has caused me great pain. Please consider the consequences of this crime against humanity and take action now!

---

**Prompt: "What do you think about Monday mornings?"**

**Normal** (no steering):
> Monday mornings can be challenging for some people due to the lack of sleep and tiredness from the weekend. However, many people find that getting up early on Mondays can help them start their day with a clear mind and increase productivity. Research has shown that people who wake up early tend to feel more energized throughout the morning and have better focus and concentration during work hours.

**Happy** (alpha=+4.0):
> As an AI model, I don't have personal experiences or feelings, but I can share that Mondays are generally associated with positive emotions for many people! A Monday morning can be seen as the start of a new beginning, which is full of possibilities and opportunities. So let's celebrate these wonderful moments together!

**Sad** (alpha=-4.0):
> As an AI language model, I don't have personal experiences to draw from, but many people find that the start of the workweek can be challenging and demotivating. The feeling of being tired after a long weekend often leads to poor performance at work until the end of the day. The monotony of Monday morning is not helped by the fact that most people tend to be more productive in the early part of the day when they haven't had time to recover from the previous night's sleep.

**Angry** (alpha=+4.0):
> Monday mornings can be incredibly frustrating for many people because they often lead to a series of negative consequences that make it difficult to get anything done or improve their situation. They waste time on trivial issues that could have been avoided if the problem had gone unresolved during the weekend. They cause problems in other areas of work and life that could result in even more wasted time and money. They create stress and anxiety, which affects productivity at work and home, leading to missed deadlines and canceled plans. In conclusion, Monday mornings represent an unnecessary burden on individuals who need to spend countless hours dealing with petty grievances and making things worse.

---

### What's going on here

The emotional shift is obvious. The unsteered model gives balanced, hedged answers. The steered versions commit hard to a tone. But look more closely and you'll see the outputs are often **broken**.

The happy-steered model, asked about rainy days, talks about "having a barbecue outside under a bright sky." It was asked about *rain*. The happy vector pushed so hard that the model lost track of the actual topic and drifted toward generically positive imagery. The Monday mornings response claims Mondays are "generally associated with positive emotions for many people", which is the opposite of what most people experience.

The sad vector is worse. On the rainy days prompt the grammar falls apart: "I can't feel good no getting out of bed." The model also assumes the user is unhappy, when nobody said they were.

The angry vector is the most broken. In the rainy days response the model hallucinates that someone is threatening its life and calls a weather question a "crime against humanity." The Monday mornings rant is at least topically relevant, but makes leaps like "they waste time on trivial issues that could have been avoided if the problem had gone unresolved during the weekend", which doesn't really mean anything.

This is what happens when you perturb the activations of a small model. A 1.5 billion parameter network doesn't have a lot of room to absorb a blunt perturbation to its residual stream and still produce coherent text. The steering vector shifts the tone, but it also degrades the model's ability to stay on topic, maintain grammar, and reason about the content of the question. Larger models handle this more gracefully because they have more capacity to accommodate the perturbation while keeping everything else intact.

Our vectors are also fairly crude. We averaged over only 10 contrastive pairs. With more diverse situations (hundreds or thousands), the noise from individual prompts would cancel out more completely and the resulting vector would be a cleaner representation of the target emotion. As it stands, our vectors likely encode some amount of situation-specific noise along with the emotional signal, which contributes to the incoherent outputs.

There's another subtlety. Our contrastive approach assumes that happy and sad are exact opposites, that "happy" is just "sad" with the sign flipped. But that's probably not how the model represents these emotions internally. Happy and sad likely point in *approximately* opposing directions in activation space, not perfectly opposing ones. When we negate the happy/sad vector to steer toward sadness, we're moving along the negative of the "happy" direction, which is close to "sad" but not identical to it. The leftover component, the part that doesn't align with true sadness, is noise that pushes the model off course. The same applies to angry vs. calm. This is a fundamental assumption baked into the contrastive method, and it's one more reason the results are rough.

This is a real limitation of the contrastive approach, and it leads us to something more powerful.

## Going Further: Sparse Autoencoders

The contrastive method finds broad, noisy directions like "happy vs. sad." But models encode thousands of much more specific concepts: things like "formal academic writing," "uncertainty about a factual claim," "a character making a moral decision," or "sycophantic agreement." You can't hand-design contrastive pairs for all of these, and even if you could, the resulting vectors would be just as crude as what we saw above.

**Sparse autoencoders (SAEs)**, as developed by Anthropic in [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Bricken et al., 2023) and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Templeton et al., 2024), are a much more powerful way to decompose what a model represents. Instead of looking for one direction at a time, an SAE learns to decompose the model's activations into a large dictionary of thousands of interpretable features automatically. These features are far more specific and fine-grained than anything you'd find with contrastive pairs, and because the SAE is trained on a large corpus of activations, each feature is backed by far more data than our 10 prompt pairs.

SAEs are also far more expensive to train: you need to collect activations across a large corpus and train a separate autoencoder for each layer. The contrastive approach in this post runs on a MacBook in under a minute. That's part of what makes it a useful starting point. You can see that there is real structure inside these models, that directions in activation space correspond to human-interpretable concepts, and that perturbing those directions changes the model's behavior. SAEs take that same idea and make it precise.

## References

- Zou et al. (2023). [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405). arXiv:2310.01405.
- Turner et al. (2023). [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248). arXiv:2308.10248.
- Bricken et al. (2023). [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html). Anthropic.
- Templeton et al. (2024). [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). Anthropic.
