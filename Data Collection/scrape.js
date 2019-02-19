var tmi = require("tmi.js")
var fs = require('fs');

// change this line to add channels
//var channelss = {"#summit1g":"0", "#imaqtpie":"1", "#riot games":"2", "#syndicate":"3",
//                 "#shroud":"4", "#esl_csgo":"5", "#ninja":"6", "#sodapoppin":"7", "#loltyler1":"8",
//                 "#gosu":"9",};
var channels = {"Jordyx3":"0","xChangas":"1","CrankUK":"2","Nathanalpizar":"3", "Mulisious" :"4"};
// create list of channels
var channellist = Array(channelss.length).fill("");
for (var key in channelss)
    channellist[channelss[key]] = key;

console.log(channellist);

var options = {
    options: {
        debug: false
    },
    connection: {
        reconnect: true
    },
    identity: {
        username: "dip_7777",
        password: "oauth:r1btqctlqag9nmuuzt36a2o96jk4wz"
    },
    channels: channellist,
};

var client = new tmi.client(options);

client.connect();

// a = client.getChannels();
// b = client.getOptions();

var totalstring = Array(channellist.length).fill("");
var counterarr = Array(channellist.length).fill(0);

var times = 0;

client.on("chat", function (channel, userstate, message, self) {
    // Don't listen to my own messages..
    if (self) return;
    var newmsg = message.replace(/[^\w\s]/gi, '');
    message = "{\"t\":\"" + userstate['tmi-sent-ts'] + "\",\"u\":\"" + userstate.username + "\",\"e\":\"" + JSON.stringify(userstate.emotes) + "\",\"m\":\"" + message + "\",\"nm\":\"" + newmsg + "\"}\n"
    totalstring[parseInt(channelss[channel])] += message;
    // console.log(channel, channelss[channel], counterarr, counterarr[parseInt(channelss[channel])]);
    // console.log(parseInt(channelss[channel]));

    times = userstate['tmi-sent-ts'];

    counterarr[parseInt(channelss[channel])]++;
    if(counterarr[parseInt(channelss[channel])] % 10 == 0)
    {
    	fs.appendFile(channellist[parseInt(channelss[channel])]+"database.txt", totalstring[parseInt(channelss[channel])], function(err) {});
        counterarr[parseInt(channelss[channel])] = 0;
        totalstring[parseInt(channelss[channel])] = "";
    }
});

var joinstring = Array(channellist.length).fill("");
client.on("join", function (channel, username, self) {
    // Do your stuff.
    if (self) return;
    // console.log(username);

    joinstring[parseInt(channelss[channel])] = "{\"t\":\"" + times + "\",\"u\":\"" + username + "\"}\n";
    fs.appendFile(channellist[parseInt(channelss[channel])]+"join.txt", joinstring[parseInt(channelss[channel])], function(err) {});
});

var partstring = Array(channellist.length).fill("");
client.on("part", function (channel, username, self) {
    // Do your stuff.
    if (self) return;
    // console.log(username);
   
    partstring[parseInt(channelss[channel])] = "{\"t\":\"" + times + "\",\"u\":\"" + username + "\"}\n";
    fs.appendFile(channellist[parseInt(channelss[channel])]+"part.txt", partstring[parseInt(channelss[channel])], function(err) {});
});

