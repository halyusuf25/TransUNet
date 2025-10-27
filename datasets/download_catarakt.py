import os
import synapseclient
import synapseutils

dest = "/data/shared/CataractData/semantic_segmentation"
os.makedirs(dest, exist_ok=True)

syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc2MTU1MjEzNiwiaWF0IjoxNzYxNTUyMTM2LCJqdGkiOiIyNzcwNSIsInN1YiI6IjM1MDc2MDMifQ.Tb7lMFbrW3w7kqqPV9kv-DCDRK6-dAGAt8vJVDi6MyITIioEDJb-x7qWT84J_KtnStQ5-MvjeE57nrvVYPPJ66pcWukUCYlGf7LOAy5exhAy8F-6xsuKnrwwoJev4UiObKTyZraQVImYRzJZJl1skCZKmlIsajjL7pB82HSOHOtX6CjL13uaP7x5t4rBJaezGhzu6Jg95RV6jBtoWKutV_-NX4mVNmewfSjOB3qo6kMW76sReraPzAARp7RU9kl5ZTJ7i9LMa6OY_nBS1aaybtLvnOjhC8hZQ92HUdglpv1yOZrULlnP6hJDDF1CfbjJSkJH_9bAAp7tLWpa-BmwQg")

files = synapseutils.syncFromSynapse(
    syn,
    "syn53395479",
    path=dest,                 # <— download here
    ifcollision="overwrite.local"    # or "overwrite.local"/"keep.local"
)
