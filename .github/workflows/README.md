# GitHub Actions workflows

## Workflow Overview

This GitHub Actions workflow consists of:

1. Triggering Conditions: Runs on pushes to the main branch, version tags `(v*)`, and pull requests.
2. Build Process: Installs dependencies, sets up environments, and builds the Tauri app.
3. Testing: Runs basic tests to validate the application.
4. Release Process: Creates a GitHub release and uploads the binaries.

**Note:** It is essential to consider that all the virtual machines are run in the root folder of the repository. Ensure that 

## Workflow Breakdown

### Triggering Conditions

```
on:
  push:
    branches:
      - main
    tags:
      - 'v*' # Trigger on version tags
  pull_request:
    branches:
      - main
  workflow_dispatch:
```
This ensures the pipeline runs on key events like code pushes, PRs, and version tagging.

### Job Strategy

```
strategy:
  fail-fast: false
  matrix:
    include:
      - platform: 'macos-latest'                # macos with m chips
        args: '--target aarch64-apple-darwin'
      - platform: 'macos-13'                    #intel macos
        args: '--target x86_64-apple-darwin'
      - platform: 'ubuntu-latest'
        args: ''
      - platform: 'windows-latest'
        args: ''
```
Using a matrix strategy allows parallel builds across different OS environments.

### Setting Up Dependencies

```
- name: Install dependencies (Ubuntu only)
  if: matrix.platform == 'ubuntu-latest'
  run: |
    sudo apt-get update
    sudo apt-get install -y libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf
```
Each platform might need specific dependencies before running builds. The above are system dependencies that are only required for the ubuntu/debian builds.

### Setting Up Rust and Node.js

```
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: lts/*
    cache: 'npm'
    # ensure that the path to the required package is correct
    cache-dependency-path: ./tauri-monitoring/package.json

- name: Install Rust Toolchain
  uses: dtolnay/rust-toolchain@stable
  with:
        # These targets are only used for macos runners so it's in an `if` statement 
        # It slightly speeds up the windows and linux builds.
        targets: ${{ matrix.platform == 'macos-latest' && 'aarch64-apple-darwin,x86_64-apple-darwin' || '' }}
```
Rust is required for Tauri builds, and Node.js is used for dependency management. Some additional arguements can be added with the `with` argument etc. to reduce build time and acquiring other functionalities based on requirements.

### Implement Caching for Faster Builds

Using cache can significantly reduce build time:
```
- name: Rust cache
  uses: swatinem/rust-cache@v2
  with:
    workspaces: './tauri-monitoring/src-tauri -> ./tauri-monitoring/src-tauri/target'
```
### Building and Testing

```
- name: Install dependencies
  working-directory: ${{env.working-directory}}
  run: npm install

- name: Test Tauri app
  working-directory: ${{env.working-directory}}
  **# run: npm test**
  run: echo "Running Tauri tests..."

- name: Build Tauri app
  working-directory: ${{env.working-directory}}
  run: npm run tauri build
```
This ensures the app is built correctly before deployment. The `run` command works like a subprocess on the command line on your operating system.

**Note 1:** When you run a script through `npm`, either as `npm run-script <name>` or with a defined shortcut like `npm test` or `npm start`, your current package directory's bin directory is placed at the front of your path. As the Current file structure for the Tauri app is a folder in the root repository, therefore the `working-directory` argument was changes and needs to be changed accordingly.

**Note 2:** There are currently no test scripts for the tauri montoring app. As soon as they are included in the repository, uncomment the **bolded** `run` command under **Test Tauri app** and uncomment the `echo` command.

### Creating a GitHub Release
```
- name: Publish Release with Binaries
  id: create_release
  uses: tauri-apps/tauri-action@v0
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  with:
    projectPath: ./tauri-monitoring
    tagName: v__VERSION__
    releaseName: 'v__VERSION__'
    releaseBody: 'See the assets to download this version and install.'
    releaseDraft: false
    prerelease: false
    args: ${{ matrix.args }}
```
This step automatically generates a GitHub release with the built Tauri app binaries.