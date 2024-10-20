import logging
from llama_index.readers.github import GithubClient, GithubRepositoryReader

def load_github_documents(github_token, owner, repo):
    github_client = GithubClient(github_token=github_token, verbose=True)
    branches = ["main", "master"]

    for branch in branches:
        try:
            documents = GithubRepositoryReader(
                github_client=github_client,
                owner=owner,
                repo=repo,
                use_parser=True,
                verbose=False,
                filter_file_extensions=(
                [
                    ".rs",
                    ".ms",
                    ".toml",
                ],
                GithubRepositoryReader.FilterType.INCLUDE,
                ),
            ).load_data(branch=branch)
            logging.info(f"GH Loaded data from branch '{branch}' for repo '{owner}/{repo}'")
            return documents
        except Exception as e:
            logging.error(f"Failed to load data from branch '{branch}' for repo '{owner}/{repo}': {e}")
    
    logging.warning(f"Skipping repo '{owner}/{repo}' as both 'main' and 'master' branches failed.")
    return None
