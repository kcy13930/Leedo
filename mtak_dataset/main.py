from dset1 import obj1
from dset2 import obj2
import json


i = 0
while i < 300:
  node_1 = obj2[i]
  node_2 = json.dumps(node_1)
  node_3 = json.loads(node_2)
  node_4 = "None"
  if node_3.get("user_profile") != None:
     node_4 = node_3.get("user_profile").get("real_name")
  #print(node_4)
  #print(node_3.get("ts"))
  print(node_3.get("text"))
  i = i + 1
