import axios from "axios";
import qs from 'qs';
import { base_url } from "../config"

export class Service {

  extractParagraph(text, callback) {
    const create_extraction_url = `${base_url}/extract`

    const data = qs.stringify({ 'paragraphs': [text] })

    axios.post(create_extraction_url, data)
      .then(function (response) {
        callback(response)
      }).catch(function (error) {
        console.log(error)
        callback()
      })
  }

}