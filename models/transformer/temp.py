from ray.util.client import ray

context = ray.init()
print(context.dashboard_url)
#%%
