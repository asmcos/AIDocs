'use strict';

/**
 *  ebook controller
 */

const { createCoreController } = require('@strapi/strapi').factories;

const fetch = require('node-fetch');
const YAML  = require('yaml');
const fs    = require('fs')
const path  = require('path')
var exec = require("child_process").exec;


var host = "http://127.0.0.1:1337"
//var host = "http://119.91.153.63:1337"

// bookid
var bookid = 1
var filelist = []
// AIDocs/ oWriter/src/api/ebook/controllers
var father = path.resolve(__dirname,'../../../../')
var bookpath = path.resolve(father,'../tmp')
var bookdocpath = path.resolve(bookpath,'docs')

function parse_nav(n){
    var val = Object.values(n)[0]

    // filename
    if(typeof val === 'string'){
        console.log(val)
        filelist.push(val)
    } else{
        //二级子目录，需要再次循环
        val.forEach(function(item){
            parse_nav(item)
        })
    }

}

async function make_filecontent(filename){

        const file_parse = path.parse(filename)

        const res = await fetch(host+"/api/file-mds?filters[filename][$eq]="+filename+"&filters[ebook][id][$eq]="+bookid)
        const datas = await res.json()

        if (datas.meta.pagination.total == 0) return

        let content = datas.data[0].attributes.content

        //console.log(file_parse.dir,file_parse.base)
        let filepath = path.resolve(bookdocpath,file_parse.dir)
        console.log(filepath)

        //如果目录不存在创建目录,支持创建多级目录
        if (!fs.existsSync(filepath)) fs.mkdirSync(filepath,{ recursive: true });

        fs.writeFileSync(path.resolve(filepath,file_parse.base),content,{flag:"w+"})

        return
}



async function bookupdate(ctx){
        // 读取书的信息
        const res = await fetch(host+"/api/ebooks")
        var datas = await res.json()
        var objdata = ""
        for (var i = 0; i < datas.data.length;i++){
                  var data = datas.data[i]
                  if(data.id == bookid){
                       objdata = data
                       break
                  }
        }
        if (objdata == "") return

        var buffer = objdata.attributes.yml
        var name   = objdata.attributes.name
        var folder = objdata.attributes.folder

        let config = YAML.parse(buffer);
        var nav = config['nav']

        if (folder != ""){
             bookpath = path.resolve(father,'../'+folder)
             bookdocpath = path.resolve(bookpath,'docs')
        }

        //创建mkdocs 项目配置文件
        if (!fs.existsSync(bookpath)) fs.mkdirSync(bookpath,{ recursive: true });
        fs.writeFileSync(path.resolve(bookpath,'mkdocs.yml'),buffer,{flag:"w+"})

        nav.forEach(function(item){
            parse_nav(item)
        })

        //创建所有的markdown 文件
        console.log("---------------------------")
        filelist.forEach(async function(item){
           await make_filecontent(item)
        })


        exec("cd "+ bookpath +";mkdocs build",function(err,stdout,stderr){
            if(err){
                console.error(err);
            }
            console.log("stdout:",stdout)
            console.log("stderr:",stderr);
        });

}


module.exports = createCoreController('api::ebook.ebook',({ strapi }) =>  ({
  // Method 1: Creating an entirely custom action
  async booksync(ctx) {

	bookupdate(ctx)

	return "ebook ok"
  }

}));
