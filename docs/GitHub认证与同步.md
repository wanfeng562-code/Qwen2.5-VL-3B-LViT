# 云服务器 GitHub 认证与同步

适用仓库：https://github.com/wanfeng562-code/Qwen2.5-VL-3B-LViT

## 认证方式

使用当前仓库专用的 Ed25519 部署密钥，通过 ssh.github.com 的 443 端口连接。GitHub 仓库的 Deploy keys 中需要启用 Allow write access。

私钥位于云服务器 /root/.ssh/id_ed25519_qwen_lvit，公钥位于同名 .pub 文件。私钥不放进项目，不进入 Git。

GitHub 主机 Ed25519 指纹已对照官方文档核验：

    SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU

known_hosts_qwen_lvit 保存已核验的主机公钥。Git 的 core.sshCommand 仅在本仓库配置，指定该密钥、严格主机校验及批处理模式，不影响其他仓库或 SSH 登录配置。

## 日常操作

在云服务器终端进入项目：

    cd /root/autodl-tmp/Qwen2.5-VL-3B-LViT
    git status
    git pull --ff-only

修改后选择代码和文档提交：

    git add <需要提交的文件>
    git commit -m "说明本次修改"
    git push

实际图像、标注、权重、缓存、输出及凭据由 .gitignore 排除。不要强制添加这些文件。

## 故障判断

- Permission denied (publickey)：检查仓库是否添加了当前公钥。
- 认证成功但推送被拒绝：检查部署密钥是否开启写入权限，以及远程分支是否出现新提交。
- SSH 认证测试可能显示认证成功后以状态码 1 退出，因为 GitHub 不提供交互式 shell；以成功提示和实际 git 操作为准。
- 远程历史出现新提交时，先检查差异，再合并；不使用强制推送覆盖他人修改。

## 密钥生命周期

部署密钥只作用于此仓库。服务器更换或不再使用时，可从该仓库 Settings → Deploy keys 移除对应密钥。

说明依据：[GitHub 部署密钥文档](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)及[官方主机指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)。
