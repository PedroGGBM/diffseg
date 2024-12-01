#Copyright 2023 Google LLC

#Use of this source code is governed by an MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.

# utilities
import tensorflow as tf
import copy
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from collections import defaultdict

# profiling
import cProfile
import pstats
import io

class DiffSeg:
  def __init__(self, kl_threshold, refine, num_points):
    # Generate the  gird
    self.grid = self.generate_sampling_grid(num_points)
    # Inialize other parameters 
    self.kl_threshold = np.array(kl_threshold)
    self.refine = refine

  def generate_sampling_grid(self,num_of_points):
    """
      - Intuition:
          The grid helps identify key points on an image where calculations will be performed.
          This speeds up processing with losing mucnh detail by focusing on specific pixels rather than
          every pixel individually.

      - Maths
          Uses linear spacing and mesh grid generation to distribute points evenly across the image.
    """

    segment_len = 63 // (num_of_points - 1)
    total_len = segment_len * (num_of_points - 1)
    start_point = (63 - total_len) // 2
    x_new = np.linspace(start_point, total_len + start_point, num_of_points)
    y_new = np.linspace(start_point, total_len + start_point, num_of_points)
    x_new, y_new = np.meshgrid(x_new, y_new, indexing='ij')
    points = np.concatenate(([x_new.reshape(-1,1), y_new.reshape(-1,1)]), axis=-1).astype(int)
    return points

  def get_weight_ratio(self, weight_list):
    """
      - Intuition:
          To fairly combine several different-sizes images/feature maps, it calculates a "weight"
          foc each based on size, ensuring bigger maps don't overshadow smaller ones.

      - Maths: 
          Compute the square root of the area (last dimension size) for each map, then normalize them
          so they sum to one, giving a proporitional influence in further processing.
    """
    
    # This function assigns proportional aggergation weight 
    sizes = []
    for weights in weight_list:
      sizes.append(np.sqrt(weights.shape[-2]))
    denom = np.sum(sizes)
    
    return sizes / denom
  
    # SUGGESTED IMPROVEMENT (np arr)
    # size = np.array([np.sqrt(w.shape[-2]) for w in weight_list])
    # return size / size.sum()

  def aggregate_weights(self, weight_list, weight_ratio=None):
    """
      - Intuition:
          Aggregates (combines) multiple attention maps into one unified map, allowing for better
          understanding of where important features are locates across different layers of a NN.

      - Maths:
          Average across channels to simplify the data.
          Upsample smaller maps to standard size (64x64) using bilinear interpolation, which smooths out values
          by considering surrounding pixels.
          Normalize each resultant map so that the sum of all values equals one.
    """

    if weight_ratio is None:
      weight_ratio = self.get_weight_rato(weight_list)
    
    aggre_weights = np.zeros((64,64,64,64), dtype=np.float32)
   
    for index, weights in enumerate(weight_list):
      size = int(np.sqrt(weights.shape[-1]))
      ratio = int(64/size)
      
      # Average over the multi-head channel
      weights = weights.mean(0).reshape(-1, size, size)

      # Upsample the last two dimensions to 64 x 64
      weights = tf.keras.layers.UpSampling2D(size=(ratio, ratio), data_format="channels_last", interpolation='bilinear')(tf.expand_dims(weights,axis=-1))
      weights = tf.reshape(weights,(size,size,64,64))

      # SUGGESTED IMPROVEMENT
      # weights = tf.image.resize(weights, [64, 64], method='bilinear')
      # weights /= tf.reduce_sum(weights, axis=(1, 2), keepdims=True)

      # Normalize to make sure each map sums to one
      weights = weights/tf.math.reduce_sum(weights,(2,3),keepdims=True)
      
      # Spatial tiling along the first two dimensions
      weights = tf.repeat(weights,repeats=ratio,axis=0)
      weights = tf.repeat(weights,repeats=ratio,axis=1)

      # Aggrgate accroding to weight_ratio
      aggre_weights += weights*weight_ratio[index]

    return aggre_weights.numpy().astype(np.double)

  def aggregate_x_weights(self, weight_list, weight_ratio=None):
    # x_weights: 8 x size**2 x 77
    # return 512 x 512 x 77
    if weight_ratio is None:
      weight_ratio = self.get_weight_rato(weight_list)
    aggre_weights = np.zeros((512, 512, 77))

    for index,weights in enumerate(weight_list):
      size = int(np.sqrt(weights.shape[-2]))
      ratio = int(512/size)
      weights = weights.mean(0).reshape(1,size,size,-1)

      # TODO: optimization -> optimize bilinear interpolation approach
      weights = tf.keras.layers.UpSampling2D(size=(ratio, ratio), data_format="channels_last", interpolation='bilinear')(weights)
      weights = weights/tf.math.reduce_sum(weights,axis=-1,keepdims=True)
      aggre_weights += weights*weight_ratio[index]
    return aggre_weights.numpy().astype(np.double)

  def KL(self,x,Y):
      """
        - Intuition:
            A measure ow how one probability distribution (set of attention values) diverges from another.
            If two distributions are very similar, their KL divergence is small; if they're different, it's large.

        - Maths:
            Compute the logarithmic difference between two sets of probabilities/densities.
            Multiply this by the original porbabilities and sum over all dimensions.

            [ for distributions (P) and (Q): 
              
              KL( P || Q ) = \sum P(x) \log\left(\frac{P(x)}{Q(x)}\right)
            
            ]

            It's symmetric here, meaning it considers how both directions of comparison differ, 
            owing to this custom implementation of KL.
      """

      # TODO: optimization -> vectorize operations for KL divergence
      qoutient = tf.math.log(x)-tf.math.log(Y)
      kl_1 = tf.math.reduce_sum(tf.math.multiply(x, qoutient),axis=(-2,-1))/2
      kl_2 = -tf.math.reduce_sum(tf.math.multiply(Y, qoutient),axis=(-2,-1))/2
      return tf.math.add(kl_1,kl_2)

  def mask_merge(self, iter, attns, kl_threshold, grid=None):
    """
      - Intuition:
          Combining similar regions into larger masks or segments.

      - Maths:
          Use KL divergence to determine similarity.
          For the first iteration, compare each "anchor" point to others and aggregate those similar enough.
          In subsequent iterations, continue reducing and refining the number of masks by aggregating close matches.
    """

    if iter == 0:

      # TODO: optimization -> reduce number of similarity checks
      # The first iteration of merging
      anchors = attns[grid[:,0],grid[:,1],:,:] # 256 x 64 x 64
      anchors = tf.expand_dims(anchors, axis=(1)) # 256 x 1 x 64 x 64
      attns = attns.reshape(1,4096,64,64) 
      # 256 x 4096 x 64 x 64 is too large for a single gpu, splitting into 16 portions
      split = np.sqrt(grid.shape[0]).astype(int)
      kl_bin=[]

      # TODO: optimization -> parallelize operation for KL divergence
      for i in range(split):
        temp = self.KL(tf.cast(anchors[i*split:(i+1)*split],tf.float16),tf.cast(attns,tf.float16)) < kl_threshold[iter] # type cast from tf.float64 to tf.float16
        kl_bin.append(temp)
      kl_bin = tf.cast(tf.concat(kl_bin, axis=0), tf.float64) # 256 x 4096
      new_attns = tf.reshape(tf.matmul(kl_bin,tf.reshape(attns,(-1,4096)))/tf.math.reduce_sum(kl_bin,1,keepdims=True),(-1,64,64)) # 256 x 64 x 64
    else:
      # The rest of merging iterations, reducing the number of masks
      matched = set()
      new_attns = []
      for i,point in enumerate(attns):
        if i in matched:
          continue
        matched.add(i)
        anchor = point
        kl_bin = (self.KL(anchor,attns) < kl_threshold[iter]).numpy() # 64 x 64
        if kl_bin.sum() > 0:
          matched_idx = np.arange(len(attns))[kl_bin.reshape(-1)]
          for idx in matched_idx: matched.add(idx)
          aggregated_attn = attns[kl_bin].mean(0)
          new_attns.append(aggregated_attn.reshape(1,64,64))
    return np.array(new_attns)

  def generate_masks(self, attns, kl_threshold, grid):
    """
      - Intuition:
          Fine-tune the merged masks for better clarity and distinction using clustering, which
          groups similar data points together.

      - Maths:
          Apply the KMeans algorithm to partition attention maps into distinct clusters based on
          similarity.
          Recalculate centroids of these clusters for more consistent segment boundaries.
    """
    
    # Iterative Attention Merging
    for i in range(len(kl_threshold)):
      if i == 0:
        # TODO: optimization -> implement caching for repeated computations in mark_merge
        attns_merged = self.mask_merge(i, attns, kl_threshold, grid=grid)
      else:
        attns_merged = self.mask_merge(i, attns_merged, kl_threshold)
    attns_merged = attns_merged[:,0,:,:]

    # Kmeans refinement (optional for better visual consistency)
    if self.refine:
      attns = attns.reshape(-1,64*64)
      kmeans = KMeans(n_clusters=attns_merged.shape[0], init=attns_merged.reshape(-1,64*64), n_init=1).fit(attns)
      clusters = kmeans.labels_
      attns_merged = []
      for i in range(len(set(clusters))):
        cluster = (i == clusters)
        attns_merged.append(attns[cluster,:].mean(0).reshape(64,64))
      attns_merged = np.array(attns_merged)

    # Upsampling
    # TODO: optimization -> optimize bilinear interpolation approach
    self.upsampled = tf.keras.layers.UpSampling2D(size=(8, 8), data_format="channels_last", interpolation='bilinear')(tf.expand_dims(attns_merged,axis=-1))

    # Non-Maximum Suppression
    M_final = tf.reshape(tf.math.argmax(self.upsampled,axis=0),(512,512)).numpy()

    return M_final
  
  # def segment(self, weight_32_, weight_32, weight_16, weight_8, weight_ratio = None):
  #   M_list = []
  #   for i in range(len(weight_32)):
  #     # Step 1: Attention Aggregation
  #     weights = self.aggregate_weights([weight_32[i],weight_32[i], weight_16[i], weight_8[i]],weight_ratio=weight_ratio)
  #     # Step 2 & 3: Iterative Merging & NMS
  #     M_final = self.generate_masks(weights, self.kl_threshold, self.grid)
  #     M_list.append(M_final)
  #   return np.array(M_list)

  def segment(self, weight_64, weight_32, weight_16, weight_8, weight_ratio = None):
    """
      - Intuition (also for get_semantics()):
          Enlarge the masks to the image resolution, then select the most prominent mask at each location.

      - Maths:
          Upsample via bilinear interpolation.
          Apply Non-Maximum Suppresion (NMS) to keep the strongest response (most confident region/mask)
          per pixel location.
    """

    M_list = []
    for i in range(len(weight_64)):
      # Step 1: Attention Aggregation
      weights = self.aggregate_weights([weight_64[i],weight_32[i], weight_16[i], weight_8[i]],weight_ratio=weight_ratio)
      # Step 2 & 3: Iterative Merging & NMS
      M_final = self.generate_masks(weights, self.kl_threshold, self.grid)
      M_list.append(M_final)
    return np.array(M_list)

  def get_semantics(self, pred, x_weight, nouns, voting="majority"):
        # This function assigns semantic labels to masks 
        indices = [item[0]+1 for item in nouns] # Igonore the first BOS token
        prompt_list = [item[1] for item in nouns]
        x_weight = x_weight[:,:,indices] # size x size x N
        x_weight = x_weight.reshape(512*512,-1)
        norm = np.linalg.norm(x_weight,axis=0,keepdims=True)
        x_weight = x_weight/norm # Normalize the cross-attention maps spatially
        pred = pred.reshape(512*512,-1)

        label_to_mask = defaultdict(list)
        for i in set(pred.flatten()):
          if voting == "majority":
            logits = x_weight[(pred==i).flatten(),:]
            index = logits.argmax(axis=-1)
            category = prompt_list[int(np.median(index))]
          else:
            logit = x_weight[(pred==i).flatten(),:].mean(0)
            category = prompt_list[logit.argmax(axis=-1)]
          label_to_mask[category].append(i)
        return label_to_mask

def profile_segmentor():
    # Initialize the DiffSeg instance with sample parameters
    diff_seg = DiffSeg(kl_threshold=[0.9]*3, refine=True, num_points=16);

    # Sample weights for each layer (should be replaced with actual data)
    weight_64 = [np.random.rand(8, 4096) for _ in range(5)]
    weight_32 = [np.random.rand(8, 1024) for _ in range(5)]
    weight_16 = [np.random.rand(8, 256) for _ in range(5)]
    weight_8  = [np.random.rand(8, 64) for _ in range(5)]

    # Begin profiling
    pr = cProfile.Profile()
    pr.enable()

    # Call the method you want to profile
    # TODO: profile and identity bottlenecks
    _ = diff_seg.segment(weight_64, weight_32, weight_16, weight_8)

    # Stop profiling
    pr.disable()

    # Output the profiling results
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_status(sortby)

    ps.print_stats()
    print(s.getValue())