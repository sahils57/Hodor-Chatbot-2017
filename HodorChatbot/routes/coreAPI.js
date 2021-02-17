/**
 * Created by paulngouchet on 6/23/17.
 */
let LocalStrategy    = require('passport-local').Strategy;
let FacebookStrategy = require('passport-facebook').Strategy;
let PythonShell      = require('python-shell');



const passport = require('passport')

const apiConfig = require('../config/apiCredentials')

const express = require('express')
const router = express.Router()



router.post('/translate', function (req, res, next) {

    var options = {
        //pythonPath: './temp/',
        scriptPath: '../routes/chatbot',
        args: [req.body.message]
    };

    PythonShell.run('response.py', options, function (err, results) {
        if (err) throw err;
        // results is an array consisting of messages collected during execution
        console.log(results);
        res.send(results)
    });

})




module.exports = router ;

