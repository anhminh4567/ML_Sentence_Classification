# Shared Dense Layers in Multi-Output Models

## What's Happening in Your Current Code

```python
# Your current implementation
for i in range(PREDICT_SIZE):
    dense1 = keras.layers.Dense(units=64, activation="relu")(flatten)
    output = keras.layers.Dense(units=len(ALL_CHARACTERS), activation="sigmoid", name=f"out_{i}")(dense1)
    outputs.append(output)
```

**What this creates:**
- 5 **separate** Dense layers, each with their **own weights**
- `dense1` is created **inside the loop**, so you get 5 different dense layers
- Each position learns independently
- Total params: 5 × (10400→64 + 64→36) = 5 × 667,004 = **3,335,020 params**

Look at your model summary:
```
│ dense_30 (Dense)    │ (None, 64)        │    665,664 │  <- Position 0
│ dense_31 (Dense)    │ (None, 64)        │    665,664 │  <- Position 1
│ dense_32 (Dense)    │ (None, 64)        │    665,664 │  <- Position 2
│ dense_33 (Dense)    │ (None, 64)        │    665,664 │  <- Position 3
│ dense_34 (Dense)    │ (None, 64)        │    665,664 │  <- Position 4
```
Five separate `Dense` layers with different weights!

---

## What Shared Dense Layers Mean

```python
# Shared implementation
dense1 = keras.layers.Dense(units=64, activation="relu")  # ← Created ONCE outside loop
dense_out = keras.layers.Dense(units=len(ALL_CHARACTERS), activation="softmax")  # ← Created ONCE

outputs = []
for i in range(PREDICT_SIZE):
    d = dense1(flatten)  # ← REUSING the same layer
    out = dense_out(d)
    outputs.append(out)
```

**What this creates:**
- 1 shared Dense layer used 5 times
- **Same weights** applied to all 5 positions
- Each call `dense1(flatten)` uses the **same** Dense layer instance
- Total params: Just 10400→64 + 64→36 = **667,004 params** (5× reduction!)

The `i` index doesn't appear in the dense layer itself - that's the point! The **same transformation** is applied 5 times to create 5 outputs.

---

## Visual Comparison

### Your Current Approach (Separate Weights):
```
                    ┌──> dense_30 ──> out_0
                    │
                    ├──> dense_31 ──> out_1
                    │
flatten (10400) ────┼──> dense_32 ──> out_2
                    │
                    ├──> dense_33 ──> out_3
                    │
                    └──> dense_34 ──> out_4
```
5 different dense layers = 5× parameters

### Shared Approach (Same Weights):
```
                    ┌──> dense_shared ──> out_0
                    │         ↓
                    ├──> dense_shared ──> out_1  (same weights!)
                    │         ↓
flatten (10400) ────┼──> dense_shared ──> out_2  (same weights!)
                    │         ↓
                    ├──> dense_shared ──> out_3  (same weights!)
                    │         ↓
                    └──> dense_shared ──> out_4  (same weights!)
```
1 dense layer reused 5× = fewer parameters

---

## How Keras Knows to Reuse Weights

In Keras, when you create a layer **once** and call it **multiple times**, it reuses the same weights:

```python
# Create layer once
my_dense = keras.layers.Dense(64)

# Use it multiple times
x1 = my_dense(input_1)  # First call: creates weights
x2 = my_dense(input_2)  # Second call: REUSES same weights
x3 = my_dense(input_3)  # Third call: REUSES same weights
```

Versus:

```python
# Create layer inside loop
for i in range(3):
    my_dense = keras.layers.Dense(64)  # NEW layer each time!
    x = my_dense(input)
```

---

## Why Would You Want This?

### Pros of sharing:
- **5× fewer parameters** → less overfitting risk
- Forces the model to learn a "general character detector"
- Works well when all positions have similar difficulty (like CAPTCHA)
- Better for small datasets (like your 700 training samples)

### Cons of sharing:
- Less flexible - can't specialize per position
- If position 0 is always harder, sharing might hurt

---

## Which Should You Use?

For **CAPTCHA with 700 training samples**:
- **Use shared layers** → You have way too many parameters for your dataset size
- Your current approach has 3.3M params for 700 images - that's a recipe for overfitting
- Shared approach would have ~670k params - much more reasonable

---

## Implementation Example

```python
def build_model_shared():
    input = keras.layers.Input(shape=IMAGE_SHAPE)

    # Convolutional layers
    convo1 = keras.layers.Convolution2D(filters=8, kernel_size=(3,3), padding="same", activation="relu")
    pooling1 = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")
    convol2 = keras.layers.Convolution2D(filters=16, kernel_size=(3,3), padding="same", activation="relu")
    pooling2 = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")

    pipeline_up_to_flatten = pooling2(convol2(pooling1(convo1(input))))
    flatten = keras.layers.Flatten()(pipeline_up_to_flatten)

    # Shared dense layers (created ONCE)
    dense1 = keras.layers.Dense(units=64, activation="relu")
    dense_out = keras.layers.Dense(units=len(ALL_CHARACTERS), activation="softmax")

    # Reuse same layers for all positions
    outputs = []
    for i in range(PREDICT_SIZE):
        d = dense1(flatten)
        output = dense_out(d)
        # Note: Keras will auto-name outputs, but we can't use custom names with shared layers
        outputs.append(output)

    model = keras.Model(inputs=input, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

---

## Key Takeaway

The `i` index doesn't appear in the dense layer because we're **intentionally reusing** the same layer. The loop just connects the same layer to the flatten output multiple times, creating multiple output branches that all share the same learned weights.