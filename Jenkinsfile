pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/nathaa13/RestAPI_ML.git']])
            }
        }
        
        stage('DockerBuild') {
            steps {
                bat "docker build -t mlops ."
            }
        }
        stage('DockerRun') {
            steps {
                bat "docker run --rm mlops"
            }
        }
        stage('Test') {
            steps {
                bat "python test_api.py "
            }
        }


    }
}