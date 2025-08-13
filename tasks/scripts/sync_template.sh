source ./tasks/shims/install_backend

template_url="https://bitbucket.org/tonkintaylor/python-template"
copier_version="9.4.1"

# check whether the .copier/.copier-answers.yml already exists, if so we should update
# otherwise just call copier directly
if [ -f ".copier/.copier-answers.yml" ]
then
    if ! uvx --python 3.12 copier@$copier_version update "git+$template_url" --trust
    then
        echo "Error: Could not update from the template repository"
        exit
    fi
else
    if [ -f "README.md" ]; then
        rm README.md
    fi
    if ! uvx --python 3.12 copier@$copier_version copy "git+$template_url" . --trust
    then
        echo "Error: Could not copy the template repository"
        exit
    fi
fi
