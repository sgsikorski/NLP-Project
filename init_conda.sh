# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/apps/software/standard/core/anaconda/2023.07-py3.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/apps/software/standard/core/anaconda/2023.07-py3.11/etc/profile.d/conda.sh" ]; then
        . "/apps/software/standard/core/anaconda/2023.07-py3.11/etc/profile.d/conda.sh"
    else
        export PATH="/apps/software/standard/core/anaconda/2023.07-py3.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<