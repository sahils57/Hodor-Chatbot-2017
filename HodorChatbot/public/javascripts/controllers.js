

let app = angular.module('aitraveller', ['ngRoute', 'ngCookies'])


// Main controller for the entire app

    app.controller('chat', function($scope, $http, $timeout, $window) {


        $scope.text = "";               // Text entered by the user to interact with the chabot

        $scope.ENGtext = "THERE";          // Translated version of the text into english



         $scope.logout = function() {           // Logging out of the app

                let openUrl = '/auth/logout/'

                window.location.replace(openUrl)


            }



        $scope.reloading = function()          // Reloading the app in case something goes wrong or also to reset the google Map
        {

            $window.location.reload()
        }



        $scope.insert = function () {



            const requestDb = {
                method: 'post',
                url: 'http://localhost:3000/api/saving',
                data: {
                    name: "traveller",
                    message: $scope.text
                }
            }


            $http(requestDb)                    // Calling API.AI with the english translation
                .then(function (response) {




                    const tensorflowResponse = {
                        method: 'post',
                        url: 'http://localhost:3000/smart/translate',
                        data: {
                            language: "en",
                            message: $scope.text                // needs to be saved in the database
                        }
                    }


                    $http(tensorflowResponse)
                        .then(function (response) {


                            $scope.ENGtext = response.data         // Response from response.py needs to be saved

                            const botDb = {
                                method: 'post',
                                url: 'http://localhost:3000/api/saving',
                                data: {
                                    name: "chabot",
                                    message: $scope.ENGtext    //ENGtext
                                }
                            }

                            $http(botDb)                    // Calling API.AI with the english translation
                                .then(function (response) {
                                    $scope.travellers = response.data



                                    $http.get('/api/db')                // retrieving all the dialog from the database and saving it in a json object rawRecords
                                        .then(function (response) {

                                            $scope.rawRecords = response.data


                                        })


                                })

                        })


                })


        }


})



    app.controller('facebook', function($scope, $http) {        // controller of the facebook logging

        $scope.facebookLog = function() {

            let openUrl = '/authenticated'

            window.location.replace(openUrl)

        }



    })


