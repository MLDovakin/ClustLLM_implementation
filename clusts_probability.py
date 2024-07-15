from scipy.spatial.distance import cdist

def get_prob_summ(target_emb, cl_d, alpha):

  closes_summ = []

  for cl, emb in cl_d.items():

    centroid_emb = np.mean(emb, axis=0) 

    diff_e = cdist([target_emb], [centroid_emb], metric='euclidean')[0]

    closes_summ.append( ((1 + diff_e**2)/alpha)**(-((alpha+1)/2) ) )

  closes_summ = sum(closes_summ)

  return closes_summ

def get_probs(target_emb, clust_emb, cl_d, alpha):

  centroid_clust = np.mean(clust_emb, axis=0)
  diff_e = np.linalg.norm(target_emb- centroid_clust)

  closest_p =  ((1 + diff_e**2)/alpha)**(-((alpha+1)/2) )

  closes_summ = get_prob_summ( target_emb, cl_d, alpha)

  p_ik =  closest_p / closes_summ
  return p_ik

sampling_clusts = {}
alpha = 1
n = 0
for data_point in range(len(df)) :

  data_point_name = df['Cluster'][data_point]
  point_embedding  = df['Embeddings'][data_point]


  prob_list = []
  point_id  = 0

  for prob_clust, prob_emb in cl_d.items():

    point_id  += 1

    prob_summ = get_probs(point_embedding, prob_emb, cl_d, alpha)
    prob_list.append((prob_clust, prob_summ.tolist()[0]))

    #print(data_point_name, point_id, prob_clust, prob_summ)
  #print({str(data_point)+ '_' + str(data_point_name): prob_list })
  #n +=1
  sampling_clusts.update({str(point_id) + '_df_ind_'+ str(data_point)+ '_clust_' + str(data_point_name): prob_list })
