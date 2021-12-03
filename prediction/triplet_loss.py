import tensorflow as tf

def _pairwise_distances(embeddings, squared=False):
  """Compute the 2D matrix of distances between all the embeddings.

  Args:
      embeddings: tensor of shape (batch_size, embed_dim)
      squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.

  Returns:
      pairwise_distances: tensor of shape (batch_size, batch_size)
  """
  # Get the dot product between all embeddings
  # shape (batch_size, batch_size)
  im_embeddings = embeddings[:, :int(embeddings.shape[1] / 2)]
  #im_embeggings_np = im_embeddings.numpy()
  anchor_emb = embeddings[:, int(embeddings.shape[1] / 2):]
  anchor_emb = tf.expand_dims(anchor_emb[0], axis=0)
  #anchor_emb_np = anchor_emb.numpy()
  dot_product = tf.matmul(im_embeddings, tf.transpose(anchor_emb))
  #dot_product_np = dot_product.numpy()
  # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
  # This also provides more numerical stability (the diagonal of the result will be exactly 0).
  # shape (batch_size,)
  square_norm_a = tf.reduce_sum(tf.square(im_embeddings), axis=1, keepdims=True)
  #square_norm_a_np = square_norm_a.numpy()
  square_norm_b = tf.reduce_sum(tf.square(anchor_emb), axis=1, keepdims=True)
  #square_norm_b_np = square_norm_b.numpy()
  # Compute the pairwise distance matrix as we have:
  # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
  # shape (batch_size, batch_size)
  distances = tf.add(square_norm_a, square_norm_b - 2.0*dot_product)

  # Because of computation errors, some distances might be negative so we put everything >= 0.0
  distances = tf.maximum(distances, 0.0)

  if not squared:
      # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
      # we need to add a small epsilon where distances == 0.0
      mask = tf.cast(tf.equal(distances, 0.0), float)
      distances = distances + mask * 1e-16

      distances = tf.sqrt(distances)

      # Correct the epsilon added: set the distances on the mask to be exactly 0.0
      distances = distances * (1.0 - mask)
  #distances_np = distances.numpy()
  return distances

def batch_hard_triplet_loss(y_true, y_pred):
  """Build the triplet loss over a batch of embeddings.

  For each anchor, we get the hardest positive and hardest negative to form a triplet.

  Args:
      labels: labels of the batch, of size (batch_size,)
      embeddings: tensor of shape (batch_size, embed_dim)
      margin: margin for triplet loss
      squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.

  Returns:
      triplet_loss: scalar tensor containing the triplet loss
  """
  # Get the pairwise distance matrix
  
  #margin = 1.
  labels = y_true
  squared=False
  labels = tf.cast(labels, dtype='int32')
  #label_np = labels.numpy()
  embeddings = y_pred

  pairwise_dist = _pairwise_distances(embeddings, squared=squared)
  #pairwise_dist_np = pairwise_dist.numpy()
  # For each anchor, get the hardest positive
  # First, we need to get a mask for every valid positive (they should have same label)
  #mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
  #mask_anchor_positive = tf.cast(mask_anchor_positive, float)

  # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
  anchor_positive_dist = tf.multiply(tf.cast(labels, float), pairwise_dist)
  #anchor_positive_dist_np=anchor_positive_dist.numpy()
  # shape (batch_size, 1)
  hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=0)

  tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

  # For each anchor, get the hardest negative
  # First, we need to get a mask for every valid negative (they should have different labels)
  #mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
  #mask_anchor_negative = tf.cast(mask_anchor_negative, float)

  # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
  #max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
  anchor_negative_dist = tf.multiply(1-(tf.cast(labels, float)), pairwise_dist)
  #anchor_negative_dist_np = anchor_negative_dist.numpy()
  # shape (batch_size,)
  zero = tf.constant(0, dtype=tf.float32)
  where = tf.not_equal(anchor_negative_dist, zero)
  hardest_negative_dist = tf.reduce_min(anchor_negative_dist[where], axis=0)
  D = hardest_positive_dist - hardest_negative_dist
  margin = tf.math.log(1 + tf.math.exp(D))
  tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

  # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
  triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

  # Get final mean triplet loss
  #triplet_loss = tf.reduce_mean(triplet_loss)

  return triplet_loss


def compute_accuracy(y_true, y_pred):   
    

  labels = y_true
  squared = False
  labels = tf.cast(labels, dtype='int32')
  margin = 1.
  embeddings = y_pred

  pairwise_dist = _pairwise_distances(embeddings, squared=squared)

  # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
  anchor_positive_dist = tf.multiply(tf.cast(labels, float), pairwise_dist)
  zero = tf.constant(0, dtype=tf.float32)
  where = tf.not_equal(anchor_positive_dist, zero)
  positive_non_zero = anchor_positive_dist[where]
  # shape (batch_size, 1)

  # For each anchor, get the hardest negative
  # First, we need to get a mask for every valid negative (they should have different labels)

  # We add the maximum value in each row to the invalid negatives (label(a) == label(n))

  anchor_negative_dist = tf.multiply(1 - (tf.cast(labels, float)), pairwise_dist)

  # shape (batch_size,)
  zero = tf.constant(0, dtype=tf.float32)
  where = tf.not_equal(anchor_negative_dist, zero)
  hardest_negative_dist = tf.reduce_min(anchor_negative_dist[where], axis=0)

  positive_less_negative = tf.less_equal(positive_non_zero, hardest_negative_dist - margin)
  positive_less_negative = tf.cast(positive_less_negative, float)
  accuracy = tf.reduce_mean(positive_less_negative)
  return accuracy